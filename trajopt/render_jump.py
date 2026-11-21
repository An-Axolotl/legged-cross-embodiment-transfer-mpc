"""Render an optimised jump with the robot's full visual meshes.

Pinocchio does the kinematics: the robot's URDF visual geometry is loaded into a pin.GeometryModel (reduced
exactly like the optimisation model for Talos) and every mesh is placed with the joint placements that
pin.forwardKinematics gives for each knot of the Crocoddyl solution.  pyrender only rasterises the posed meshes
(offscreen, EGL).

For the "--same" runs the optimisation stretched the thigh and shank segments to 0.38 / 0.33 m; each limb mesh is
stretched along its own segment axis by the same ratio, so the drawn robot matches the optimised kinematics.

    PYOPENGL_PLATFORM=egl python render_jump.py go1 [--same] [--fwddyn] --res results/jump --out media
outputs  <out>/videos/jump_<tag>.mp4  and  <out>/gifs/jump_<tag>.gif
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
np.infty = np.inf  # pyrender 0.1.45 still uses the alias removed in NumPy 2
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from jump import ERD, ROBOTS, _seg, rescale  # noqa: E402


def load_with_geometry(name):
    """Same model as jump.load(), plus the matching visual GeometryModel."""
    spec = ROBOTS[name]
    urdf = os.path.join(ERD, spec["urdf"])
    m = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
    g = pin.buildGeomFromUrdf(m, urdf, pin.GeometryType.VISUAL, package_dirs=[os.path.dirname(os.path.dirname(ERD)), ERD])
    pin.loadReferenceConfigurations(m, os.path.join(ERD, spec["srdf"]), False)
    qf = m.referenceConfigurations[spec["ref"]].copy()
    if "keep" not in spec:
        return m, g, qf, spec
    lock = [m.getJointId(n) for n in m.names[2:] if n not in spec["keep"]]
    mr, gr = pin.buildReducedModel(m, g, lock, qf)  # upper body frozen at the reference posture
    q0 = pin.neutral(mr)
    q0[:7] = qf[:7]
    for n in spec["keep"]:
        q0[mr.joints[mr.getJointId(n)].idx_q] = qf[m.joints[m.getJointId(n)].idx_q]
    return mr, gr, q0, spec


def segment_stretch(m0, m1, spec):
    """parent joint id -> 3x3 stretch (in that joint's frame) mapping original to rescaled segment lengths."""
    S = {}
    for segs in (spec["thigh"], spec["shank"]):
        for a, b in segs:
            kind, cid = _seg(m0, a, b)
            if kind == "joint":
                pj, t0, t1 = m0.parents[cid], m0.jointPlacements[cid].translation, m1.jointPlacements[cid].translation
            else:
                pj, t0, t1 = m0.frames[cid].parentJoint, m0.frames[cid].placement.translation, m1.frames[cid].placement.translation
            d = t0 / np.linalg.norm(t0)
            r = np.linalg.norm(t1) / np.linalg.norm(t0)
            S[pj] = np.eye(3) + (r - 1.0) * np.outer(d, d)
    return S


def primitive(geom):
    """trimesh for a coal/hppfcl primitive shape, None for meshes."""
    k = type(geom).__name__
    if k == "Box":
        return trimesh.creation.box(2 * np.asarray(geom.halfSide))
    if k == "Sphere":
        return trimesh.creation.icosphere(radius=geom.radius, subdivisions=3)
    if k == "Cylinder":
        return trimesh.creation.cylinder(radius=geom.radius, height=2 * geom.halfLength, sections=32)
    if k == "Capsule":
        return trimesh.creation.capsule(radius=geom.radius, height=2 * geom.halfLength, count=[24, 24])
    return None


def build_meshes(g, stretch):
    """One pyrender mesh per geometry object, vertices expressed in the parent joint frame."""
    out = []
    for go in g.geometryObjects:
        tm = primitive(go.geometry)
        if tm is None:
            if not go.meshPath or not os.path.exists(go.meshPath):
                print("skip", go.name)
                continue
            tm = trimesh.load(go.meshPath, force="mesh")
            tm.vertices = np.asarray(tm.vertices) * np.asarray(go.meshScale)
        v = np.asarray(tm.vertices)
        P = go.placement
        v = v @ P.rotation.T + P.translation
        if go.parentJoint in stretch:
            v = v @ stretch[go.parentJoint].T
        tm = trimesh.Trimesh(vertices=v, faces=tm.faces, process=False)
        c = np.asarray(go.meshColor)
        if np.allclose(c[:3], c[0]) and c[0] > 0.85:  # no colour in the URDF: neutral dark grey
            c = np.array([0.25, 0.27, 0.30, 1.0])
        else:  # URDF colour, slightly darkened for contrast against the light floor
            c = np.r_[c[:3] * 0.75, c[3]]
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=c, metallicFactor=0.2, roughnessFactor=0.6)
        out.append((go.parentJoint, pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False)))
    return out


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", choices=list(ROBOTS))
    ap.add_argument("--same", action="store_true")
    ap.add_argument("--fwddyn", action="store_true")
    ap.add_argument("--mass", type=float, default=9.07)
    ap.add_argument("--thigh", type=float, default=0.38)
    ap.add_argument("--shank", type=float, default=0.33)
    ap.add_argument("--effort", type=float, default=80.0)
    ap.add_argument("--dt", type=float, default=1e-2)
    ap.add_argument("--res", default="results/jump")
    ap.add_argument("--out", default="media")
    ap.add_argument("--size", type=int, nargs=2, default=(720, 450))
    a = ap.parse_args()

    tag = f"{a.robot}{'_same' if a.same else ''}{'_fwd' if a.fwddyn else ''}"
    m0, g, q0, spec = load_with_geometry(a.robot)
    m = m0.copy()
    stretch = {}
    if a.same:
        m = rescale(m, spec, a.mass, a.thigh, a.shank, a.effort)
        stretch = segment_stretch(m0, m, spec)
    xs = np.load(os.path.join(a.res, f"{tag}_tau.npz"), allow_pickle=True)["xs"]
    assert xs.shape[1] == m.nq + m.nv, (xs.shape, m.nq, m.nv)
    d = m.createData()
    meshes = build_meshes(g, stretch)

    # scene: ground with 25 cm stripes, start / target lines
    scene = pyrender.Scene(bg_color=[0.96, 0.97, 0.99, 1.0], ambient_light=[0.35, 0.35, 0.35])
    x0 = xs[0][0]
    for i, x in enumerate(np.arange(x0 - 4.0, x0 + 5.0, 0.25)):
        tile = trimesh.creation.box((0.25, 30.0, 0.02))
        tile.apply_translation((x + 0.125, 10.0, -0.01))
        shade = 0.62 if i % 2 else 0.70
        scene.add(pyrender.Mesh.from_trimesh(tile, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[shade, shade, shade, 1.0], roughnessFactor=0.9)))
    for xl, col in [(x0, [0.2, 0.45, 0.8, 1]), (x0 + 1.0, [0.85, 0.3, 0.2, 1])]:
        line = trimesh.creation.box((0.02, 2.0, 0.004))
        line.apply_translation((xl, 0, 0.002))
        scene.add(pyrender.Mesh.from_trimesh(line, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=col)))
    nodes = [(j, scene.add(mesh)) for j, mesh in meshes]
    com_node = scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(radius=0.025), material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.84, 0.15, 0.16, 1])))

    zmax = max(float(xs[:, 2].max()) + 0.5, 1.0)
    target = np.array([x0 + 0.5, 0.0, zmax * 0.45])
    eye = target + np.array([0.35, -3.4, 0.55])
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(38), aspectRatio=a.size[0] / a.size[1]), pose=look_at(eye, target))
    scene.add(pyrender.DirectionalLight(intensity=3.5), pose=look_at(target + [-1.5, -2.0, 3.0], target))
    scene.add(pyrender.DirectionalLight(intensity=1.2), pose=look_at(target + [2.0, 1.0, 2.0], target))
    r = pyrender.OffscreenRenderer(*a.size)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    frames = []
    for k, x in enumerate(xs):
        q = x[:m.nq]
        pin.forwardKinematics(m, d, q)
        for j, node in nodes:
            scene.set_pose(node, d.oMi[j].homogeneous)
        T = np.eye(4)
        T[:3, 3] = pin.centerOfMass(m, d, q)
        scene.set_pose(com_node, T)
        col, _ = r.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        im = Image.fromarray(col)
        ImageDraw.Draw(im).text((12, 10), f"{tag}   t = {k * a.dt:.2f} s   (rendered from the Pinocchio model)",
                                fill=(30, 30, 30), font=font)
        frames.append(np.asarray(im))
    r.delete()
    os.makedirs(f"{a.out}/videos", exist_ok=True)
    os.makedirs(f"{a.out}/gifs", exist_ok=True)
    fps = int(round(1 / a.dt / 2))  # half speed, as before
    imageio.mimsave(f"{a.out}/videos/jump_{tag}.mp4", frames + [frames[-1]] * fps, fps=fps, macro_block_size=1)
    small = [np.asarray(Image.fromarray(f).resize((360, 225), Image.LANCZOS)) for f in frames[::2]]
    imageio.mimsave(f"{a.out}/gifs/jump_{tag}.gif", small + [small[-1]] * 12, duration=2000 / fps, loop=0)
    print("wrote", tag, len(frames), "frames")


if __name__ == "__main__":
    main()
