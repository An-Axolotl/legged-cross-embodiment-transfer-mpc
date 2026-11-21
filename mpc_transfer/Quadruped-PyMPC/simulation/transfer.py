"""Cross-robot controller transfer (headless, reproducible).

Two modes:

  record   : run the NMPC on the SOURCE robot (default Go1) with a fixed forward
             velocity command and save, at every control step, the quantities that
             the transfer re-uses on the target robot:
               - reference base linear / angular velocity
               - leg bias forces h(q, v)  (Coriolis + centrifugal + gravity)
               - leg mass matrices M_leg(q)
               - centroidal (base) inertia
               - base position trajectory (used as the tracking target)

  transfer : run the NMPC on a TARGET robot, but feed it the SOURCE robot's
             recorded references and dynamics terms at every step (the original
             `is_transfer = True` branch of simulation.py). The target's own
             kinematics (feet Jacobians, feet positions, base state) are still
             measured from the target in MuJoCo. Tracking error against the
             source base trajectory is accumulated in x, y and z.

Robot selection is done with the XFER_ROBOT environment variable, because
quadruped_pympc.config is evaluated at import time:

    XFER_ROBOT=go1 python simulation/transfer.py record   --out runs/go1_source.npz
    XFER_ROBOT=b2  python simulation/transfer.py transfer --src runs/go1_source.npz --video media/go1_to_b2.mp4
"""
import argparse
import json
import os
import time

import mujoco
import numpy as np

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr
from quadruped_pympc import config as cfg
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper

LEGS = ["FL", "FR", "RL", "RR"]
PARTS = ("ref", "inertia", "legM", "bias")

# "rich" command profile (t [s], vx, vy [m/s], yaw rate [rad/s]); linearly interpolated.
# Speed changes, turns both ways and a sidestep, so that the dynamics model actually matters.
RICH_PROFILE = np.array([
    [0.0, 0.0, 0.0, 0.0], [2.0, 0.4, 0.0, 0.0], [6.0, 0.4, 0.0, 0.0], [7.0, 0.4, 0.0, 0.4],
    [10.0, 0.4, 0.0, 0.4], [11.0, 0.2, 0.15, 0.0], [13.0, 0.2, 0.15, 0.0], [14.0, 0.5, 0.0, -0.4],
    [17.0, 0.5, 0.0, -0.4], [20.0, 0.0, 0.0, 0.0]])


def profile_cmd(t):
    P = RICH_PROFILE
    vx, vy, wz = (np.interp(t, P[:, 0], P[:, i]) for i in (1, 2, 3))
    return np.array([vx, vy, 0.0]), np.array([0.0, 0.0, wz])


def make_env(fwd_vel, friction):
    env = QuadrupedEnv(robot=cfg.robot,
                       hip_height=cfg.hip_height,
                       legs_joint_names=cfg.robot_leg_joints,
                       feet_geom_name=cfg.robot_feet_geom_names,
                       scene=cfg.simulation_params['scene'],
                       sim_dt=cfg.simulation_params['dt'],
                       ref_base_lin_vel=fwd_vel,       # fixed float -> constant command
                       ref_base_ang_vel=0.0,
                       ground_friction_coeff=friction,
                       base_vel_command_type="forward",
                       state_obs_names=('base_pos',))
    if cfg.qpos0_js is not None:
        env.mjModel.qpos0 = np.concatenate((env.mjModel.qpos0[:7], cfg.qpos0_js))
    env.reset(random=False)
    # Recompute kinematics for the reset state. Without this, geom/site positions stay at the
    # values computed for model.qpos0, which differ from the standing keyframe when cfg.qpos0_js
    # is set (Mini Cheetah): the controller was then initialised with wrong foot positions.
    mujoco.mj_forward(env.mjModel, env.mjData)
    return env


def perturb_dynamics(env, mass_scale, seed):
    """Randomise link masses/inertias of the target (used for the 'Go1 perturbed' transfer)."""
    rng = np.random.default_rng(seed)
    m = env.mjModel
    for b in range(1, m.nbody):
        s = mass_scale if b == 1 else rng.uniform(0.8, 1.2)
        m.body_mass[b] *= s
        m.body_inertia[b] *= s


class Recorder:
    def __init__(self, env, path, fps=30, w=640, h=480):
        self.path, self.fps = path, fps
        self.r = mujoco.Renderer(env.mjModel, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = 1
        self.cam.distance = max(1.2, 6.0 * cfg.hip_height)
        self.cam.elevation = -20
        self.cam.azimuth = 135
        self.frames, self.last_t = [], -1.0

    def maybe_grab(self, env):
        t = env.mjData.time
        if t - self.last_t >= 1.0 / self.fps:
            self.r.update_scene(env.mjData, self.cam)
            self.frames.append(self.r.render().copy())
            self.last_t = t

    def save(self):
        import imageio
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        imageio.mimsave(self.path, self.frames, fps=self.fps, codec="libx264", quality=7)


def run(mode, steps, fwd_vel, friction, src=None, out=None, video=None, mass_scale=None, seed=0, action_noise=2.0,
        noise_frac=None, profile='forward', parts=PARTS, scale=None):
    scale = scale or {}
    np.random.seed(seed)
    env = make_env(fwd_vel, friction)
    if mass_scale is not None:
        perturb_dynamics(env, mass_scale, seed)

    sim_dt = cfg.simulation_params['dt']
    tau_limits = LegsAttr(**{l: env.mjModel.actuator_ctrlrange[env.legs_tau_idx[l]] * 0.9 for l in LEGS})
    jac_prev = LegsAttr(*[np.zeros((3, env.mjModel.nv)) for _ in range(4)])
    tau = LegsAttr(*[np.zeros((env.mjModel.nv, 1)) for _ in range(4)])
    wrapper = QuadrupedPyMPC_Wrapper(initial_feet_pos=env.feet_pos, legs_order=LEGS,
                                     quadrupedpympc_observables_names=('ref_base_height', 'phase_signal'))
    rec = Recorder(env, video) if video else None

    if mode == "transfer":
        S = dict(np.load(src))  # load once (NpzFile re-reads on every index)
        n_src = len(S["ref_lin"])
        steps = min(steps, n_src)

    log = {k: [] for k in ["ref_lin", "ref_ang", "qfrc", "mass", "inertia", "base_pos"]}
    err = np.zeros(3)
    err_hist = []
    fell_at = None
    t0 = time.time()

    for k in range(steps):
        feet_pos = env.feet_pos(frame='world')
        hip_pos = env.hip_positions(frame='world')
        base_lin_vel = env.base_lin_vel(frame='world')
        base_ang_vel = env.base_ang_vel(frame='world')
        base_ori = env.base_ori_euler_xyz
        base_pos = env.base_pos.copy()
        ref_lin, ref_ang = env.target_base_vel()
        if profile == 'rich':
            ref_lin, ref_ang = profile_cmd(env.mjData.time)

        inertia = (env.get_base_inertia().flatten() if cfg.simulation_params['use_inertia_recomputation']
                   else cfg.inertia.flatten())
        qpos, qvel = env.mjData.qpos, env.mjData.qvel
        joints_pos = LegsAttr(FL=qpos[7:10], FR=qpos[10:13], RL=qpos[13:16], RR=qpos[16:19])
        legs_mass = env.legs_mass_matrix
        legs_bias = env.legs_qfrc_bias

        feet_jac = env.feet_jacobians(frame='world', return_rot_jac=False)
        jac_dot = LegsAttr(**{l: (feet_jac[l] - jac_prev[l]) / sim_dt for l in LEGS})
        jac_prev = feet_jac
        feet_vel = LegsAttr(**{l: feet_jac[l] @ env.mjData.qvel for l in LEGS})

        if mode == "record":
            log["ref_lin"].append(np.array(ref_lin)); log["ref_ang"].append(np.array(ref_ang))
            log["qfrc"].append(np.stack([legs_bias[l] for l in LEGS]))
            log["mass"].append(np.stack([legs_mass[l] for l in LEGS]))
            log["inertia"].append(np.array(inertia)); log["base_pos"].append(base_pos)
        else:  # transfer: source robot's references and dynamics terms
            if "ref" in parts:
                ref_lin, ref_ang = S["ref_lin"][k], S["ref_ang"][k]
            if "bias" in parts:
                legs_bias = LegsAttr(**{l: S["qfrc"][k][i] * scale.get("bias", 1.0) for i, l in enumerate(LEGS)})
            if "legM" in parts:
                legs_mass = LegsAttr(**{l: S["mass"][k][i] * scale.get("legM", 1.0) for i, l in enumerate(LEGS)})
            if "inertia" in parts:
                inertia = S["inertia"][k] * scale.get("inertia", 1.0)

        tau = wrapper.compute_actions(base_pos, base_lin_vel, base_ori, base_ang_vel,
                                      feet_pos, hip_pos, joints_pos, None,
                                      LEGS, sim_dt, ref_lin, ref_ang,
                                      env.step_num, qpos, qvel, feet_jac, jac_dot, feet_vel, legs_bias,
                                      legs_mass, env.legs_qpos_idx, env.legs_qvel_idx, tau, inertia)
        for l in LEGS:
            tau[l] = np.clip(tau[l], tau_limits[l][:, 0], tau_limits[l][:, 1])
        obs = wrapper.get_obs()

        action = np.zeros(env.mjModel.nu)
        for l in LEGS:
            action[env.legs_tau_idx[l]] = tau[l]
        if noise_frac is not None:  # noise proportional to each joint's torque limit (comparable across robot sizes)
            action += np.random.normal(0, 1, size=env.mjModel.nu) * noise_frac * env.mjModel.actuator_ctrlrange[:, 1]
        else:
            action += np.random.normal(0, action_noise, size=env.mjModel.nu)  # as in the original simulation.py
        _, _, term, trunc, _ = env.step(action=action)

        if mode == "transfer":
            e = np.array([abs(base_pos[0] - S["base_pos"][k][0]),
                          abs(base_pos[1] - S["base_pos"][k][1]),
                          abs(obs["ref_base_height"] - base_pos[2])])
            err += e
            err_hist.append(e)
            log["base_pos"].append(base_pos)
        if rec:
            rec.maybe_grab(env)
        if term:
            fell_at = k * sim_dt
            # keep a short tail of the fall in the video
            for _ in range(int(0.75 / sim_dt)):
                env.step(action=np.zeros(env.mjModel.nu))
                if rec:
                    rec.maybe_grab(env)
            break

    steps_done = k + 1
    result = dict(robot=cfg.robot, mode=mode, steps=steps_done, sim_time_s=steps_done * sim_dt,
                  fell=fell_at is not None, fell_at_s=fell_at, wall_s=round(time.time() - t0, 1),
                  final_base_x=float(env.base_pos[0]), mass_scale=mass_scale,
                  action_noise=action_noise, noise_frac=noise_frac, vel_cmd=fwd_vel, profile=profile,
                  parts=list(parts), scale=scale, seed=seed)
    if mode == "transfer":
        result.update(err_x=float(err[0]), err_y=float(err[1]), err_z=float(err[2]), err_total=float(err.sum()),
                      mean_err_x_m=float(err[0] / steps_done), mean_err_y_m=float(err[1] / steps_done),
                      mean_err_z_m=float(err[2] / steps_done))
    if out and mode == "record":
        np.savez_compressed(out, **{k: np.array(v) for k, v in log.items()})
    if mode == "transfer" and out:
        # 50 Hz, float32: enough for the analysis and keeps ~900 runs small
        np.savez_compressed(out.replace(".json", "_traj.npz"), stride=10,
                            err=np.array(err_hist, np.float32)[::10], base_pos=np.array(log["base_pos"], np.float32)[::10])
    if rec:
        rec.save()
    env.close()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "transfer"])
    ap.add_argument("--steps", type=int, default=5000, help="control steps (dt = 2 ms)")
    ap.add_argument("--vel", type=float, default=0.4, help="forward velocity command [m/s]")
    ap.add_argument("--friction", type=float, default=1.5)
    ap.add_argument("--src", default="runs/go1_source.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--mass-scale", type=float, default=None, help="perturb target base mass (others +-20%%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-noise", type=float, default=2.0, help="std of Gaussian torque noise [N·m]")
    ap.add_argument("--noise-frac", type=float, default=None,
                    help="instead: noise std as a fraction of each joint's torque limit (0.044 = 2 N·m on Go1)")
    ap.add_argument("--profile", choices=["forward", "rich"], default="forward",
                    help="record mode: constant forward command, or the 20 s rich profile")
    ap.add_argument("--parts", default=",".join(PARTS), help="transfer mode: which source quantities to inject")
    for q in ("inertia", "legM", "bias"):
        ap.add_argument(f"--scale-{q}", type=float, default=None, help=f"multiply the source's {q} (sweeps)")
    a = ap.parse_args()
    scale = {q: getattr(a, f"scale_{q}") for q in ("inertia", "legM", "bias") if getattr(a, f"scale_{q}") is not None}
    res = run(a.mode, a.steps, a.vel, a.friction, a.src, a.out, a.video, a.mass_scale, a.seed, a.action_noise, a.noise_frac,
              a.profile, tuple(a.parts.split(",")), scale)
    print("RESULT " + json.dumps(res))
