"""Turn any 12-DoF quadruped URDF (e.g. from example-robot-data) into a gym-quadruped robot model.

What it keeps exactly: kinematic tree, link masses / inertias / COMs, joint limits, effort limits.
What it approximates: geometry. Meshes are dropped and replaced by capsules (legs), a box (trunk)
and spheres (feet); only the trunk/legs/feet collide with the floor, never with each other.
Joint damping / armature / friction are scaled from the Go1 model (see JOINT_* below).

Outputs, in <out>/<name>/:
  <name>.xml          MJCF with free base, legs renamed {FL,FR,RL,RR}_{hip,thigh,calf}_joint,
                      hip bodies {leg}_hip, foot geoms {leg}, torque motors, keyframe 'home'
  scene_flat.xml      flat-ground scene (same look as the gym-quadruped ones)
  <name>.urdf         geometry-free URDF with the same joint names (used by the controller's IK)
  xfer_robot.json     parameters read by quadruped_pympc/config.py (mass, hip height, ...)

    python urdf_to_gymquadruped.py --urdf a1.urdf --srdf a1.srdf --name a1 --out ../extra_robot_models
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

LEGS = ["FL", "FR", "RL", "RR"]
SEG = ["hip", "thigh", "calf"]


def strip_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    foot_radius = {}
    for link in root.findall("link"):
        for c in link.findall("collision"):
            s = c.find("geometry/sphere")
            if s is not None:
                foot_radius[link.get("name")] = float(s.get("radius"))
        for tag in ("visual", "collision"):
            for e in link.findall(tag):
                link.remove(e)
    for tag in ("gazebo", "transmission"):
        for e in root.findall(tag):
            root.remove(e)
    for j in root.findall("joint"):
        if j.get("type") == "continuous":
            j.set("type", "revolute")
            lim = j.find("limit") if j.find("limit") is not None else ET.SubElement(j, "limit")
            lim.set("lower", lim.get("lower", "-3.14")); lim.set("upper", lim.get("upper", "3.14"))
    return tree, foot_radius


def effort_limits(tree):
    out = {}
    for j in tree.getroot().findall("joint"):
        lim = j.find("limit")
        if j.get("type") == "revolute" and lim is not None and lim.get("effort"):
            out[j.get("name")] = float(lim.get("effort"))
    return out


def load_mj(tree, floating=True):
    root = ET.fromstring(ET.tostring(tree.getroot()))
    children = {j.find("child").get("link") for j in root.findall("joint")}
    base = [l.get("name") for l in root.findall("link") if l.get("name") not in children][0]
    if floating:
        ET.SubElement(root, "link", name="world")
        fj = ET.SubElement(root, "joint", name="floating_base", type="floating")
        ET.SubElement(fj, "parent", link="world"); ET.SubElement(fj, "child", link=base)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", fusestatic="false", discardvisual="true", balanceinertia="true",
                  boundmass="1e-4", boundinertia="1e-7", strippath="true")
    xml = ET.tostring(root, encoding="unicode")
    m = mujoco.MjModel.from_xml_string(xml)
    return m, base


def find_legs(m):
    """Map each foot body to its leg name and its 3 hinge joints (base -> foot order)."""
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    feet = [b for b in range(m.nbody) if "foot" in m.body(b).name.lower()]
    if len(feet) != 4:  # no foot links: use the 4 deepest leaf bodies
        leaves = [b for b in range(1, m.nbody) if not any(m.body_parentid[c] == b for c in range(m.nbody))]
        feet = sorted(leaves, key=lambda b: d.xpos[b][2])[:4]
    legs = {}
    for f in feet:
        chain, b = [], f
        while b > 1:
            for j in range(m.jnt_bodyid.size):
                if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    chain.append(j)
            b = m.body_parentid[b]
        chain = chain[::-1]
        assert len(chain) == 3, f"foot {m.body(f).name}: expected 3 hinge joints, got {len(chain)}"
        hip_body = m.jnt_bodyid[chain[0]]
        p = d.xpos[hip_body] - d.xpos[1]
        name = ("F" if p[0] > 0 else "R") + ("L" if p[1] > 0 else "R")
        legs[name] = dict(foot=f, joints=chain, hip_body=hip_body)
    assert sorted(legs) == sorted(LEGS), f"could not classify legs: {sorted(legs)}"
    return legs


def standing_pose(m, legs, srdf, state="standing"):
    """Joint angles (by original joint name) for a standing pose."""
    if srdf and os.path.exists(srdf):
        for gs in ET.parse(srdf).getroot().iter("group_state"):
            if gs.get("name") == state:
                return {j.get("name"): float(j.get("value")) for j in gs.findall("joint")
                        if len(j.get("value").split()) == 1}, f"SRDF '{state}'"  # skip the root free-flyer entry
    # no SRDF: per leg, put the foot under the hip at 80 % of the leg's full reach, knees pointing backwards
    d = mujoco.MjData(m)
    q = {}
    for leg, L in legs.items():
        jh, jt, jc = L["joints"]
        rng = lambda j: m.jnt_range[j] if m.jnt_limited[j] else np.array([-np.pi, np.pi])
        cand = []
        for t in np.linspace(*rng(jt), 91):
            for c in np.linspace(*rng(jc), 91):
                d.qpos[:] = m.qpos0
                d.qpos[m.jnt_qposadr[jt]] = t; d.qpos[m.jnt_qposadr[jc]] = c
                mujoco.mj_kinematics(m, d)
                hip = d.xpos[L["hip_body"]]
                f = d.xpos[L["foot"]] - hip
                knee = d.xpos[m.jnt_bodyid[jc]] - hip
                if knee[0] > f[0]:  # dog-like: knee behind the foot
                    continue
                cand.append((t, c, f[0], f[2]))
        cand = np.array(cand)
        reach = -cand[:, 3].min()
        k = np.argmin(cand[:, 2] ** 2 + (cand[:, 3] + 0.8 * reach) ** 2)
        q[m.joint(jh).name] = 0.0; q[m.joint(jt).name] = float(cand[k, 0]); q[m.joint(jc).name] = float(cand[k, 1])
    return q, "grid search (no SRDF): foot under hip at 80 % reach"


def composite_inertia(m, d):
    com = d.subtree_com[1]; I = np.zeros((3, 3))
    R0 = d.xmat[1].reshape(3, 3)
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3); Ib = R @ np.diag(m.body_inertia[b]) @ R.T
        r = d.xipos[b] - com
        I += Ib + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    return R0.T @ I @ R0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True); ap.add_argument("--srdf", default=None)
    ap.add_argument("--state", default="standing"); ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--default-effort", type=float, default=None, help="used when the URDF has no effort limits")
    ap.add_argument("--effort", type=float, default=None, help="override the URDF effort limit of every joint [N·m]")
    a = ap.parse_args()

    tree, foot_r = strip_urdf(a.urdf)
    efforts = effort_limits(tree)
    m, base = load_mj(tree)
    legs = find_legs(m)

    # rename joints in the URDF (so URDF, MJCF and controller share names), then reload
    rename = {}
    for leg, L in legs.items():
        for s, j in zip(SEG, L["joints"]):
            rename[m.joint(j).name] = f"{leg}_{s}_joint"
    q_stand_orig, pose_src = standing_pose(m, legs, a.srdf, a.state)
    foot_radius = {leg: foot_r.get(m.body(L["foot"]).name, None) for leg, L in legs.items()}
    hip_body_name = {m.body(L["hip_body"]).name: f"{leg}_hip" for leg, L in legs.items()}
    hip_body_name.update({m.body(L["foot"]).name: f"{leg}_foot" for leg, L in legs.items()})  # IK expects {leg}_foot links
    foot_body_name = {leg: m.body(L["foot"]).name for leg, L in legs.items()}
    for j in tree.getroot().findall("joint"):
        if j.get("name") in rename:
            j.set("name", rename[j.get("name")])
    for l in tree.getroot().iter():
        for k in ("name", "link"):
            if l.tag in ("link", "parent", "child") and l.get(k) in hip_body_name:
                l.set(k, hip_body_name[l.get(k)])
    efforts = {rename.get(k, k): v for k, v in efforts.items()}
    q_stand = {rename[k]: v for k, v in q_stand_orig.items() if k in rename}

    os.makedirs(f"{a.out}/{a.name}", exist_ok=True)
    tree.write(f"{a.out}/{a.name}/{a.name}.urdf")
    m, base = load_mj(tree)
    legs = find_legs(m)
    jn = [f"{leg}_{s}_joint" for leg in LEGS for s in SEG]
    total_mass = float(m.body_mass.sum())
    if a.effort:
        efforts = {j: a.effort for j in jn}
    if not efforts or any(e <= 0 for e in efforts.values()):
        assert a.default_effort, "URDF has no effort limits: pass --default-effort"
        efforts = {j: a.default_effort for j in jn}

    # standing pose and base height (feet on the ground)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0; d.qpos[2] = 1.0
    for j in jn:
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = q_stand.get(j, 0.0)
    mujoco.mj_forward(m, d)
    leg_len = np.mean([np.linalg.norm(d.xpos[legs[l]["foot"]] - d.xpos[legs[l]["hip_body"]]) for l in LEGS])
    r_foot = np.median([r for r in foot_radius.values() if r] or [max(0.01, 0.06 * leg_len)])
    min_foot_z = min(d.xpos[legs[l]["foot"]][2] for l in LEGS)
    base_z = 1.0 - (min_foot_z - r_foot)
    d.qpos[2] = base_z; mujoco.mj_forward(m, d)
    I = composite_inertia(m, d)

    # ---- write the MJCF ------------------------------------------------------------------------------------
    mujoco.mj_saveLastXML(f"{a.out}/{a.name}/_raw.xml", m)
    x = ET.parse(f"{a.out}/{a.name}/_raw.xml"); root = x.getroot(); os.remove(f"{a.out}/{a.name}/_raw.xml")
    root.set("model", a.name)
    for e in root.findall("compiler"):
        e.set("angle", "radian"); e.set("autolimits", "true")
        # masses/inertias come only from the URDF <inertial> tags, never from the approximate geometry added here
        e.set("inertiafromgeom", "false")
    opt = root.find("option") if root.find("option") is not None else ET.Element("option")
    opt.set("cone", "elliptic"); opt.set("impratio", "100")
    if root.find("option") is None:
        root.insert(1, opt)
    # Go1 joint params scaled with effort (damping/friction) and mass (armature)
    JOINT_DAMPING, JOINT_FRICTION, JOINT_ARMATURE = 2.0 / 23.7, 0.2 / 23.7, 0.01 / 12.7
    bodies = {b.get("name"): b for b in root.iter("body")}
    hips = np.array([d.xpos[legs[l]["hip_body"]] - d.xpos[1] for l in LEGS])
    rad = max(0.008, 0.07 * leg_len)
    col = dict(contype="1", conaffinity="0", condim="3", friction="0.8 0.02 0.01", rgba="0.25 0.25 0.28 1")
    base_body = bodies[m.body(1).name]
    ET.SubElement(base_body, "geom", name="trunk", type="box",
                  size=f"{np.abs(hips[:,0]).max()+rad:.4f} {np.abs(hips[:,1]).max():.4f} {max(0.25*np.abs(hips[:,1]).max(), 0.02):.4f}",
                  **{**col, "rgba": "0.15 0.15 0.18 1"})
    for leg in LEGS:
        L = legs[leg]
        for s, j in zip(SEG, L["joints"]):
            jel = [e for e in root.iter("joint") if e.get("name") == f"{leg}_{s}_joint"][0]
            e = efforts[f"{leg}_{s}_joint"]
            jel.set("damping", f"{JOINT_DAMPING*e:.4f}"); jel.set("frictionloss", f"{JOINT_FRICTION*e:.4f}")
            jel.set("armature", f"{JOINT_ARMATURE*total_mass:.4f}")
        # capsules: thigh body origin -> calf body origin, calf origin -> 85 % towards the foot
        jt, jc = L["joints"][1], L["joints"][2]
        tb, cb, fb = m.jnt_bodyid[jt], m.jnt_bodyid[jc], L["foot"]
        # the calf capsule is thinner than the foot and stops at 80 % so only the foot touches the ground
        for body, target, frac, nm, rr in [(tb, cb, 1.0, "thigh", rad), (cb, fb, 0.80, "calf", min(rad, 0.8 * r_foot))]:
            p = d.xpos[target] - d.xpos[body]
            p_local = d.xmat[body].reshape(3, 3).T @ p * frac
            ET.SubElement(bodies[m.body(body).name], "geom", name=f"{leg}_{nm}_geom", type="capsule",
                          fromto=f"0 0 0 {p_local[0]:.4f} {p_local[1]:.4f} {p_local[2]:.4f}", size=f"{rr:.4f}", **col)
        ET.SubElement(bodies[m.body(fb).name], "geom", name=leg, type="sphere", size=f"{r_foot:.4f}",
                      contype="1", conaffinity="0", condim="6", friction="0.8 0.02 0.01", priority="1",
                      rgba="0.9 0.55 0.1 1")
    act = ET.SubElement(root, "actuator")
    for leg in LEGS:
        for s in SEG:
            e = efforts[f"{leg}_{s}_joint"]
            ET.SubElement(act, "motor", name=f"{leg}_{s}", joint=f"{leg}_{s}_joint",
                          ctrlrange=f"{-e:.3f} {e:.3f}", forcerange=f"{-e:.3f} {e:.3f}")
    kq = d.qpos.copy(); kq[2] = base_z + 0.01
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="home", qpos=" ".join(f"{v:.5f}" for v in kq))
    ET.indent(x)
    x.write(f"{a.out}/{a.name}/{a.name}.xml")

    # scene
    scene = f"""<mujoco model="{a.name} scene">
  <include file="{a.name}.xml"/>
  <statistic center="0 0 0.1" extent="{max(0.8, 3*leg_len):.2f}"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""
    open(f"{a.out}/{a.name}/scene_flat.xml", "w").write(scene)
    info = dict(robot=a.name, source_urdf=os.path.basename(a.urdf), standing_pose=pose_src,
                mass=round(total_mass, 4), inertia=np.round(I, 6).tolist(), hip_height=round(float(base_z), 4),
                urdf_filename=f"{a.name}.urdf", leg_length=round(float(leg_len), 4), foot_radius=round(float(r_foot), 4),
                effort_limits=efforts, q_stand=q_stand, renamed_from={v: k for k, v in rename.items()})
    json.dump(info, open(f"{a.out}/{a.name}/xfer_robot.json", "w"), indent=1)
    print(json.dumps({k: info[k] for k in ["robot", "mass", "hip_height", "leg_length", "foot_radius", "standing_pose"]}))


if __name__ == "__main__":
    main()
