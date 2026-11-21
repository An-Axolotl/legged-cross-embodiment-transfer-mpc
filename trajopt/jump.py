"""Jump trajectory optimisation: same task, different embodiments.

Solves a 1 m long / 0.5 m high jump for a quadruped (ANYmal) and a humanoid
(Talos legs) with Crocoddyl on top of Pinocchio, using the inverse-dynamics
direct transcription of Ferrolho et al. (fwddyn=False, solved with SolverIntro)
or the forward-dynamics one (--fwddyn, solved with FDDP).

Phases (Crocoddyl's jumping problems):
  take-off (all feet on ground) -> fly up (reach CoM apex) -> fly down
  -> landing impulse (feet at target) -> landed (CoM at target)

"--same" rescales both robots to the same total mass, thigh / shank lengths
and torque limit, so that only the morphology (2 vs 4 legs) differs.

Outputs (in --out):
  <tag>.json       summary (convergence, cost, peak torque, total |tau|, CoM margin)
  <tag>_tau.png    per-joint torque profiles
  <tag>_tau.npz    raw trajectories
  <tag>.mp4        quick stick-figure preview (the README videos come from render_jump.py)
"""
import argparse
import json
import os

import crocoddyl
import matplotlib
import numpy as np
import pinocchio as pin

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

ERD = os.environ.get("ERD_ROBOTS", os.path.join(os.environ.get("CONDA_PREFIX", ""), "share/example-robot-data/robots"))

ROBOTS = {
    "anymal": dict(urdf="anymal_c_simple_description/urdf/anymal.urdf",
                   srdf="anymal_c_simple_description/srdf/anymal.srdf", ref="standing",
                   feet=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
                   thigh=[("LF_HFE", "LF_KFE"), ("RF_HFE", "RF_KFE"), ("LH_HFE", "LH_KFE"), ("RH_HFE", "RH_KFE")],
                   shank=[("LF_KFE", "LF_FOOT"), ("RF_KFE", "RF_FOOT"), ("LH_KFE", "LH_FOOT"), ("RH_KFE", "RH_FOOT")]),
    "go1": dict(urdf="go1_description/urdf/go1.urdf", srdf="go1_description/srdf/go1.srdf", ref="standing",
                feet=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                thigh=[(f"{l}_thigh_joint", f"{l}_calf_joint") for l in ("FL", "FR", "RL", "RR")],
                shank=[(f"{l}_calf_joint", f"{l}_foot") for l in ("FL", "FR", "RL", "RR")]),
    "b1": dict(urdf="b1_description/urdf/b1.urdf", srdf="b1_description/srdf/b1.srdf", ref="standing",
               feet=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
               thigh=[(f"{l}_thigh_joint", f"{l}_calf_joint") for l in ("FL", "FR", "RL", "RR")],
               shank=[(f"{l}_calf_joint", f"{l}_foot") for l in ("FL", "FR", "RL", "RR")]),
    "hyq": dict(urdf="hyq_description/robots/hyq_no_sensors.urdf", srdf="hyq_description/srdf/hyq.srdf", ref="standing",
                feet=["lf_foot", "rf_foot", "lh_foot", "rh_foot"],
                thigh=[(f"{l}_hfe_joint", f"{l}_kfe_joint") for l in ("lf", "rf", "lh", "rh")],
                shank=[(f"{l}_kfe_joint", f"{l}_foot") for l in ("lf", "rf", "lh", "rh")]),
    "talos": dict(urdf="talos_data/robots/talos_reduced.urdf", srdf="talos_data/srdf/talos.srdf", ref="half_sitting",
                  feet=["right_sole_link", "left_sole_link"], foot_half_len=0.1,
                  keep=["leg_left_%d_joint" % i for i in range(1, 7)] + ["leg_right_%d_joint" % i for i in range(1, 7)],
                  thigh=[("leg_left_3_joint", "leg_left_4_joint"), ("leg_right_3_joint", "leg_right_4_joint")],
                  shank=[("leg_left_4_joint", "leg_left_5_joint"), ("leg_right_4_joint", "leg_right_5_joint")]),
}


def load(name):
    spec = ROBOTS[name]
    m = pin.buildModelFromUrdf(os.path.join(ERD, spec["urdf"]), pin.JointModelFreeFlyer())
    pin.loadReferenceConfigurations(m, os.path.join(ERD, spec["srdf"]), False)
    q0 = m.referenceConfigurations[spec["ref"]].copy()
    if "keep" in spec:  # Talos: keep only the 12 leg joints (as in Crocoddyl's biped examples)
        lock = [m.getJointId(n) for n in m.names[2:] if n not in spec["keep"]]
        m = pin.buildReducedModel(m, lock, q0)
        q0 = pin.neutral(m)
        full = pin.buildModelFromUrdf(os.path.join(ERD, spec["urdf"]), pin.JointModelFreeFlyer())
        pin.loadReferenceConfigurations(full, os.path.join(ERD, spec["srdf"]), False)
        qf = full.referenceConfigurations[spec["ref"]]
        q0[:7] = qf[:7]
        for n in spec["keep"]:
            q0[m.joints[m.getJointId(n)].idx_q] = qf[full.joints[full.getJointId(n)].idx_q]
    return m, q0, spec


def _seg(m, a, b):
    """Return (child_kind, child_id) whose placement sets the length of segment a->b."""
    if m.existJointName(b):
        return "joint", m.getJointId(b)
    return "frame", m.getFrameId(b)


def rescale(m, spec, mass, thigh, shank, effort):
    """Same mass / limb lengths / torque limit for every embodiment."""
    s = mass / pin.computeTotalMass(m)
    for i in range(1, m.njoints):
        I = m.inertias[i]
        m.inertias[i] = pin.Inertia(I.mass * s, I.lever, I.inertia * s)
    for segs, L in [(spec["thigh"], thigh), (spec["shank"], shank)]:
        for a, b in segs:
            kind, cid = _seg(m, a, b)
            if kind == "joint":
                p = m.jointPlacements[cid]
                p.translation = p.translation / np.linalg.norm(p.translation) * L
                m.jointPlacements[cid] = p
            else:  # foot frame hangs off the knee joint (possibly through fixed links)
                f = m.frames[cid]
                f.placement.translation = f.placement.translation / np.linalg.norm(f.placement.translation) * L
                m.frames[cid] = f
    m.effortLimit[6:] = effort
    return m


def put_on_ground(m, q0, feet):
    d = m.createData()
    pin.framesForwardKinematics(m, d, q0)
    z = min(d.oMf[m.getFrameId(f)].translation[2] for f in feet)
    q0 = q0.copy()
    q0[2] -= z
    return q0


def solve(m, q0, spec, height, length, dt, ground, flying, fwddyn, iters):
    x0 = np.concatenate([q0, np.zeros(m.nv)])
    if len(spec["feet"]) == 4:
        from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem
        gait = SimpleQuadrupedalGaitProblem(m, *spec["feet"], fwddyn=fwddyn)
    else:
        from crocoddyl.utils.biped import SimpleBipedGaitProblem
        gait = SimpleBipedGaitProblem(m, *spec["feet"], fwddyn=fwddyn)
    prob = gait.createJumpingProblem(x0, height, [length, 0.0, 0.0], dt, ground, flying)
    solver = crocoddyl.SolverFDDP(prob) if fwddyn else crocoddyl.SolverIntro(prob)
    xs = [x0] * (prob.T + 1)
    us = prob.quasiStatic([x0] * prob.T)
    ok = solver.solve(xs, us, iters, False)
    return prob, solver, ok


def torques(m, prob, solver, dt):
    """tau_k = RNEA(q_k, v_k, a_k, f_ext,k) restricted to actuated joints."""
    d = m.createData()
    taus, base_res, contact_any = [], [], []
    for k, (rm, rd) in enumerate(zip(prob.runningModels, prob.runningDatas)):
        if not hasattr(rm, "differential") or rm.dt == 0:  # landing impulse knot: no torques
            continue
        x, xn = solver.xs[k], solver.xs[k + 1]
        q, v = x[:m.nq], x[m.nq:]
        dtk = rm.dt
        a = (xn[m.nq:] - v) / dtk
        fext = rd.differential.multibody.contacts.fext
        tau = pin.rnea(m, d, q, v, a, fext)
        taus.append(tau[6:].copy())
        base_res.append(np.linalg.norm(tau[:6]))
        contact_any.append(len(rm.differential.contacts.contacts.todict()))
    return np.array(taus), np.array(base_res), np.array(contact_any)


def com_margin(m, q, feet, half_len=0.0):
    """Signed distance of the CoM ground projection to the support region along x (+ = inside).

    Point feet for the quadruped; for the biped each sole extends +-half_len in x
    (0.2 m x 0.1 m sole, as in Crocoddyl's biped CoP support)."""
    d = m.createData()
    com = pin.centerOfMass(m, d, q)
    pin.framesForwardKinematics(m, d, q)
    fx = [d.oMf[m.getFrameId(f)].translation[0] + s for f in feet for s in (-half_len, half_len)]
    return float(min(com[0] - min(fx), max(fx) - com[0])), float(com[0]), (float(min(fx)), float(max(fx)))


def animate(m, xs, feet, path, dt, title):
    d = m.createData()
    frames = []
    for x in xs:
        q = x[:m.nq]
        pin.framesForwardKinematics(m, d, q)
        segs = []
        for j in range(2, m.njoints):
            p = m.parents[j]
            segs.append((d.oMi[p].translation.copy(), d.oMi[j].translation.copy()))
        for f in feet:
            fid = m.getFrameId(f)
            segs.append((d.oMi[m.frames[fid].parentJoint].translation.copy(), d.oMf[fid].translation.copy()))
        frames.append((segs, pin.centerOfMass(m, d, q).copy()))
    allp = np.array([p for s, _ in frames for seg in s for p in seg])
    lo, hi = allp.min(0) - 0.1, allp.max(0) + 0.1
    lo[2] = 0.0
    ext = hi - lo
    fig = plt.figure(figsize=(7.2, 7.2 * max(ext[2], 0.3) / ext[0] + 0.6))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(0, 0, 1, 0.93)

    def draw(i):
        ax.cla()
        segs, com = frames[i]
        for a, b in segs:
            ax.plot(*zip(a, b), color="#1f4e79", lw=3)
        ax.scatter(*com, color="#d62728", s=40, label="CoM")
        ax.plot([lo[0], hi[0]], [0, 0], [0, 0], color="gray", lw=1.5)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(ext)
        ax.view_init(elev=8, azim=-90)
        ax.set_axis_off()
        ax.set_title(f"{title}   t = {i * dt:.2f} s", fontsize=10)
    ani = animation.FuncAnimation(fig, draw, frames=len(frames))
    ani.save(path, writer=animation.FFMpegWriter(fps=int(round(1 / dt / 2))), dpi=100)
    plt.close(fig)


def plot_tau(taus, names, dt, path, title):
    t = np.arange(len(taus)) * dt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for j in range(taus.shape[1]):
        ax.plot(t, taus[:, j], lw=1, label=names[j])
    ax.set_xlabel("time [s]"); ax.set_ylabel("joint torque [N·m]"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=6, ncol=4); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", choices=list(ROBOTS))
    ap.add_argument("--height", type=float, default=0.5)
    ap.add_argument("--length", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--ground-knots", type=int, default=25)
    ap.add_argument("--flying-knots", type=int, default=30)
    ap.add_argument("--fwddyn", action="store_true")
    ap.add_argument("--same", action="store_true", help="rescale to --mass/--thigh/--shank/--effort")
    ap.add_argument("--mass", type=float, default=9.07, help="kg (20 lb)")
    ap.add_argument("--thigh", type=float, default=0.38)
    ap.add_argument("--shank", type=float, default=0.33)
    ap.add_argument("--effort", type=float, default=80.0, help="N·m, same limit for every joint")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default="results/jump")
    a = ap.parse_args()

    m, q0, spec = load(a.robot)
    if a.same:
        m = rescale(m, spec, a.mass, a.thigh, a.shank, a.effort)
    q0 = put_on_ground(m, q0, spec["feet"])
    prob, solver, ok = solve(m, q0, spec, a.height, a.length, a.dt, a.ground_knots, a.flying_knots, a.fwddyn, a.iters)
    taus, base_res, _ = torques(m, prob, solver, a.dt)
    margin, comx, fx = com_margin(m, solver.xs[a.ground_knots][:m.nq], spec["feet"],
                                spec.get("foot_half_len", 0.0))
    lim = m.effortLimit[6:]
    tag = f"{a.robot}{'_same' if a.same else ''}{'_fwd' if a.fwddyn else ''}"
    os.makedirs(a.out, exist_ok=True)
    res = dict(robot=a.robot, same=a.same, formulation="forward" if a.fwddyn else "inverse",
               mass_kg=round(pin.computeTotalMass(m), 2), converged=bool(ok), iterations=solver.iter,
               cost=round(float(solver.cost), 2), knots=prob.T,
               final_base_xyz=np.round(solver.xs[-1][:3], 3).tolist(),
               peak_abs_tau_Nm=round(float(np.abs(taus).max()), 1),
               total_abs_tau_Nm=round(float(np.abs(taus).sum()), 1),
               total_abs_tau_impulse_Nms=round(float(np.abs(taus).sum() * a.dt), 2),
               max_tau_over_limit=round(float((np.abs(taus) / lim).max()), 2),
               base_wrench_residual_max=float(base_res.max()),
               prelaunch_com_x=round(comx, 3), prelaunch_feet_x=[round(v, 3) for v in fx],
               prelaunch_com_margin_m=round(margin, 3))
    json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=1)
    np.savez(f"{a.out}/{tag}_tau.npz", tau=taus, xs=np.array(solver.xs), us=np.array(list(solver.us), dtype=object))
    names = [m.names[j] for j in range(2, m.njoints)]
    plot_tau(taus, names, a.dt, f"{a.out}/{tag}_tau.png",
             f"{tag}: 1 m long / {a.height} m high jump  (Σ|τ| = {res['total_abs_tau_Nm']:.0f} N·m)")
    animate(m, solver.xs, spec["feet"], f"{a.out}/{tag}.mp4", a.dt, tag)
    print("RESULT " + json.dumps(res))
