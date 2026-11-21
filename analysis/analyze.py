"""When does cross-embodiment MPC transfer work?  Reads results/{native,matrix,ablation,sweep}
and writes results/analysis/ (figures, summary.json, tables.md).

    source deps/env.sh && python analysis/analyze.py

Per robot we take the quantities that the transfer actually swaps, time-averaged over the robot's own
native run (results/native/native_<r>.npz): base inertia I (3x3, from MuJoCo's mass matrix), leg mass
matrices M_leg (4 x 3x3), leg bias forces h_leg (4 x 3, mostly gravity), plus mass, hip height and torque
limits from the model/config.  For a source s and target t we measure how wrong each injected term is
*from the target's point of view* and relate it to what happened.
"""
import glob
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = f"{ROOT}/results"
OUT = f"{RES}/analysis"
os.makedirs(OUT, exist_ok=True)
from run_experiments import ROBOTS  # noqa: E402

FAMILY = {"go1": "Unitree", "go2": "Unitree", "a1": "Unitree", "aliengo": "Unitree", "laikago": "Unitree",
          "b1": "Unitree", "b2": "Unitree", "mini_cheetah": "MIT", "anymal_b": "ANYbotics", "anymal_c": "ANYbotics",
          "hyq": "IIT", "hyqreal": "IIT", "solo12": "ODRI"}
WORKS_DRIFT = 0.25   # [m] mean planar drift from the target's own native run, below which we call a transfer "works"


# ----------------------------------------------------------------------------------------------- robot features
def model_info(r):
    """mass / hip height / torque limit / knee layout, read in a subprocess (config is import-time)."""
    code = r"""
import json, numpy as np, mujoco, os
from quadruped_pympc import config as cfg
import gym_quadruped
p = os.path.join(os.path.dirname(gym_quadruped.__file__), 'robot_model', cfg.robot, cfg.robot + '.xml')
m = mujoco.MjModel.from_xml_path(p); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
knee = {}
for leg in ['FL', 'RL']:
    j = m.joint(cfg.robot_leg_joints[leg][2]).id
    kx = d.xanchor[j][0]; fx = d.geom_xpos[m.geom(leg).id][0]
    knee[leg] = 'back' if kx < fx - 1e-3 else ('front' if kx > fx + 1e-3 else 'straight')
print(json.dumps(dict(mass_model=float(m.body_mass.sum()), mass_cfg=float(cfg.mass), hip_height=float(cfg.hip_height),
      tau_max=float(np.mean(m.actuator_ctrlrange[:, 1])), knee_front=knee['FL'], knee_rear=knee['RL'])))
"""
    p = subprocess.run(["python", "-c", code], env=dict(os.environ, XFER_ROBOT=r), capture_output=True, text=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def robot_features():
    cache = f"{OUT}/robot_features.json"
    F = json.load(open(cache)) if os.path.exists(cache) else {}
    for r in ROBOTS:
        if r in F:
            continue
        n = np.load(f"{RES}/native/native_{r}.npz")
        f = model_info(r)
        f.update(I=n["inertia"].mean(0).reshape(3, 3).tolist(), M=n["mass"].mean(0).tolist(), h=n["qfrc"].mean(0).tolist(),
                 family=FAMILY[r])
        F[r] = f
    json.dump(F, open(cache, "w"), indent=1)
    return F


def pair_features(F, s, t):
    S, T = F[s], F[t]
    Is, It = np.diag(S["I"]), np.diag(T["I"])
    Ms, Mt = np.array(S["M"]), np.array(T["M"])
    hs, ht = np.array(S["h"]), np.array(T["h"])
    return dict(
        inertia_mismatch=float(np.mean(np.abs(np.log(Is / It)))),          # mean |ln(I_s/I_t)| over the 3 axes
        inertia_ratio=float(np.exp(np.mean(np.log(Is / It)))),             # geometric-mean I_s/I_t (signed)
        legM_mismatch=float(np.linalg.norm(Ms - Mt) / np.linalg.norm(Mt)),  # relative Frobenius error
        bias_mismatch=float(np.linalg.norm(hs - ht) / np.linalg.norm(ht)),  # relative error of gravity comp.
        mass_ratio=S["mass_model"] / T["mass_model"],
        size_ratio=S["hip_height"] / T["hip_height"],
        torque_ratio=S["tau_max"] / T["tau_max"],
        knees_same=float(S["knee_front"] == T["knee_front"] and S["knee_rear"] == T["knee_rear"]),
        same_family=float(S["family"] == T["family"]),
    )


# --------------------------------------------------------------------------------------------------- outcomes
def native_xy(r):
    return np.load(f"{RES}/native/native_{r}.npz")["base_pos"][:, :2]


def outcome(path, target):
    r = json.load(open(path))
    z = np.load(path.replace(".json", "_traj.npz"))
    k = int(z["stride"]) if "stride" in z else 1
    tr = z["base_pos"][:, :2].astype(float)
    nat = native_xy(target)[::k]
    n = min(len(tr), len(nat))
    drift = float(np.mean(np.linalg.norm(tr[:n] - nat[:n], axis=1)))
    return dict(fell=bool(r["fell"]), fell_at=r["fell_at_s"], drift=drift,
                err_src_xy=r["mean_err_x_m"] + r["mean_err_y_m"], err_z=r["mean_err_z_m"])


def label(o):
    return "fell" if o["fell"] else ("works" if o["drift"] < WORKS_DRIFT else "degraded")


def main():
    F = robot_features()
    order = sorted(ROBOTS, key=lambda r: F[r]["mass_model"])
    rows = []
    for s in ROBOTS:
        for t in ROBOTS:
            p = f"{RES}/matrix/{s}_to_{t}.json"
            if os.path.exists(p):
                o = outcome(p, t)
                rows.append(dict(src=s, tgt=t, **pair_features(F, s, t), **o, label=label(o)))
    json.dump(rows, open(f"{OUT}/matrix_rows.json", "w"), indent=0)
    off = [r for r in rows if r["src"] != r["tgt"]]
    diag = {r["tgt"]: r["drift"] for r in rows if r["src"] == r["tgt"]}

    # ---------------- 1) heat map of the matrix
    D = np.full((len(order), len(order)), np.nan); L = np.empty_like(D, dtype=object)
    for r in rows:
        i, j = order.index(r["src"]), order.index(r["tgt"])
        D[i, j] = r["drift"]; L[i, j] = r["label"]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(np.log10(np.maximum(D, 1e-3)), cmap="viridis_r", vmin=-2, vmax=0.5)
    for i in range(len(order)):
        for j in range(len(order)):
            if L[i, j] == "fell":
                ax.text(j, i, "✕", ha="center", va="center", color="red", fontsize=13, fontweight="bold")
            elif L[i, j]:
                ax.text(j, i, f"{D[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if L[i, j] == "works" else "black")
    lab = [f"{r} ({F[r]['mass_model']:.0f} kg)" for r in order]
    ax.set_xticks(range(len(order)), lab, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)), lab, fontsize=8)
    ax.set_xlabel("target robot (executes)"); ax.set_ylabel("source robot (recorded model)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("log10 mean drift from target's own run [m]")
    ax.set_title(f"Transfer matrix, 20 s rich command profile. ✕ = fell. Numbers = drift [m]; works < {WORKS_DRIFT} m", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_transfer_matrix.png", dpi=150); plt.close(fig)

    # ---------------- 2) which pair features explain the outcome
    feats = ["inertia_mismatch", "legM_mismatch", "bias_mismatch", "mass_ratio", "size_ratio", "torque_ratio",
             "knees_same", "same_family"]
    def logabs(f, v):
        return abs(np.log(v)) if f in ("mass_ratio", "size_ratio", "torque_ratio") else v
    y_fail = np.array([r["label"] != "works" for r in off], float)
    y_drift = np.array([np.log10(r["drift"]) if not r["fell"] else np.log10(10.0) for r in off])
    corr, stumps = {}, {}
    for f in feats:
        x = np.array([logabs(f, r[f]) for r in off])
        rho, p = spearmanr(x, y_drift)
        corr[f] = dict(spearman_rho=round(float(rho), 3), p=float(p))
        best = (0, None, None)
        for thr in np.unique(x):
            for sign in (1, -1):
                pred = (sign * x > sign * thr)
                acc = np.mean(pred == y_fail.astype(bool))
                if acc > best[0]:
                    best = (acc, float(thr), sign)
        stumps[f] = dict(accuracy=round(best[0], 3), threshold=best[1], fails_if=">" if best[2] == 1 else "<=")
    base_rate = max(np.mean(y_fail), 1 - np.mean(y_fail))

    from sklearn.tree import DecisionTreeClassifier, export_text
    X = np.array([[logabs(f, r[f]) for f in feats] for r in off])
    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=0).fit(X, y_fail)
    rules = export_text(tree, feature_names=[f if f not in ("mass_ratio", "size_ratio", "torque_ratio") else f"|ln {f}|"
                                             for f in feats], show_weights=True)
    from sklearn.model_selection import cross_val_score
    cv = cross_val_score(DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, random_state=0), X, y_fail, cv=5)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, f, xl in [(axs[0], "inertia_mismatch", "inertia mismatch  mean |ln(I_src / I_tgt)|"),
                      (axs[1], "legM_mismatch", "leg mass-matrix mismatch  ‖M_src − M_tgt‖ / ‖M_tgt‖"),
                      (axs[2], "bias_mismatch", "bias-force mismatch  ‖h_src − h_tgt‖ / ‖h_tgt‖")]:
        for lab_, c, mk in [("works", "tab:green", "o"), ("degraded", "tab:orange", "s"), ("fell", "tab:red", "x")]:
            pts = [r for r in off if r["label"] == lab_]
            ax.scatter([r[f] for r in pts], [r["drift"] if not r["fell"] else 3.0 for r in pts], c=c, marker=mk,
                       s=22, label=lab_, alpha=0.8)
        ax.set_yscale("log"); ax.set_xscale("symlog", linthresh=0.1); ax.set_xlabel(xl, fontsize=9)
        ax.axhline(WORKS_DRIFT, ls="--", c="k", lw=0.7); ax.grid(alpha=0.3)
        ax.set_title(f"Spearman ρ = {corr[f]['spearman_rho']:+.2f}", fontsize=9)
    axs[0].set_ylabel("drift from own native run [m] (fell plotted at 3 m)"); axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_mismatch_vs_outcome.png", dpi=140); plt.close(fig)

    # ---------------- 3) ablation: which single injected component breaks it
    abl = {}
    for part in ["ref", "inertia", "legM", "bias"]:
        res = []
        for p in glob.glob(f"{RES}/ablation/{part}__*.json"):
            s, t = os.path.basename(p)[len(part) + 2:-5].split("_to_")
            o = outcome(p, t); res.append(dict(src=s, tgt=t, **o, label=label(o)))
        if res:
            abl[part] = dict(n=len(res), works=sum(r["label"] == "works" for r in res),
                             degraded=sum(r["label"] == "degraded" for r in res), fell=sum(r["fell"] for r in res),
                             median_drift=float(np.median([r["drift"] for r in res if not r["fell"]] or [np.nan])), rows=res)
    full = dict(n=len(off), works=sum(r["label"] == "works" for r in off), degraded=sum(r["label"] == "degraded" for r in off),
                fell=sum(r["fell"] for r in off), median_drift=float(np.median([r["drift"] for r in off if not r["fell"]] or [np.nan])))
    if abl:
        names = [k for k in ["ref", "inertia", "legM", "bias"] if k in abl] + ["all"]
        data = [abl[k] for k in names[:-1]] + [full]
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        x = np.arange(len(names)); b = np.zeros(len(names))
        for key, c in [("works", "tab:green"), ("degraded", "tab:orange"), ("fell", "tab:red")]:
            v = np.array([d[key] / d["n"] for d in data]); ax.bar(x, v, bottom=b, color=c, label=key); b += v
        ax.set_xticks(x, ["references only\n(own dynamics)", "+ inertia", "+ leg mass matrix", "+ bias forces", "everything"],
                      fontsize=8)
        ax.set_ylabel("fraction of the 156 cross-robot pairs"); ax.legend(fontsize=8, loc="lower left")
        ax.set_title("Which transferred component breaks the target?", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation.png", dpi=140); plt.close(fig)

    # ---------------- 4) controlled sweeps on Go1
    sweep = {}
    for p in glob.glob(f"{RES}/sweep/*.json"):
        q, k, sd = os.path.basename(p)[:-5].rsplit("_", 2)
        o = outcome(p, "go1"); sweep.setdefault(q, {}).setdefault(float(k[1:]), []).append(o)
    if sweep:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for q, c in [("inertia", "tab:blue"), ("legM", "tab:purple"), ("bias", "tab:brown")]:
            if q not in sweep:
                continue
            ks = sorted(sweep[q]); med = [np.median([o["drift"] if not o["fell"] else 3.0 for o in sweep[q][k]]) for k in ks]
            ax.plot(ks, med, "o-", c=c, label={"inertia": "base inertia", "legM": "leg mass matrix", "bias": "leg bias forces"}[q])
            for k in ks:
                if any(o["fell"] for o in sweep[q][k]):
                    ax.plot(k, 3.0, "x", c="r", ms=9)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.axhline(WORKS_DRIFT, ls="--", c="k", lw=0.7)
        ax.set_xlabel("scale applied to the injected Go1 quantity (target = Go1)"); ax.set_ylabel("median drift [m] (fell = 3 m, ✕)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig_sweeps.png", dpi=140); plt.close(fig)
    sweep_tab = {q: {str(k): dict(fell=sum(o["fell"] for o in v), n=len(v),
                                  median_drift=float(np.median([o["drift"] for o in v if not o["fell"]] or [np.nan])))
                     for k, v in sorted(d.items())} for q, d in sweep.items()}

    # ---------------- 5) families
    fam = {}
    for key, sel in [("same family", lambda r: r["same_family"]), ("different family", lambda r: not r["same_family"])]:
        sub = [r for r in off if sel(r)]
        fam[key] = dict(n=len(sub), works=sum(r["label"] == "works" for r in sub), fell=sum(r["fell"] for r in sub))
    for f_ in sorted(set(FAMILY.values())):
        sub = [r for r in off if FAMILY[r["src"]] == f_ and FAMILY[r["tgt"]] == f_]
        if sub:
            fam[f"within {f_}"] = dict(n=len(sub), works=sum(r["label"] == "works" for r in sub), fell=sum(r["fell"] for r in sub))

    # ---------------- 6) the operating envelope in signed ratios (direction matters)
    def envelope(rr, lo, hi, key="inertia_ratio"):
        inside = [r for r in rr if lo <= r[key] <= hi]; outside = [r for r in rr if not (lo <= r[key] <= hi)]
        return dict(range=[lo, hi], inside_n=len(inside), inside_works=sum(r["label"] == "works" for r in inside),
                    outside_n=len(outside), outside_works=sum(r["label"] == "works" for r in outside),
                    outside_fell=sum(r["fell"] for r in outside),
                    rule_accuracy=round(float(np.mean([(lo <= r[key] <= hi) == (r["label"] == "works") for r in rr])), 3))
    env_I = envelope(off, 0.2, 3.0)
    env_m = envelope(off, 1 / 3, 2.0, key="mass_ratio")
    bins = [0, 0.1, 0.2, 0.5, 2, 3, 5, 10, np.inf]
    def binned(rr, key):
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sub = [r for r in rr if lo <= r[key] < hi]
            if sub:
                out.append(dict(bin=f"[{lo:g}, {hi:g})", n=len(sub), works=sum(r["label"] == "works" for r in sub),
                                degraded=sum(r["label"] == "degraded" for r in sub), fell=sum(r["fell"] for r in sub)))
        return out
    abl_I = [dict(r, inertia_ratio=pair_features(F, r["src"], r["tgt"])["inertia_ratio"]) for r in abl.get("inertia", {}).get("rows", [])]
    fig, ax = plt.subplots(figsize=(6.8, 5))
    for lab_, c, mk in [("works", "tab:green", "o"), ("degraded", "tab:orange", "s"), ("fell", "tab:red", "x")]:
        pts = [r for r in off if r["label"] == lab_]
        ax.scatter([r["inertia_ratio"] for r in pts], [r["mass_ratio"] for r in pts], c=c, marker=mk, s=30, label=lab_)
    ax.axvspan(0.2, 3.0, color="tab:green", alpha=0.08); ax.axhspan(1 / 3, 2.0, color="tab:blue", alpha=0.06)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("I_source / I_target  (base inertia the target's MPC is given, relative to its own)")
    ax.set_ylabel("m_source / m_target"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"All 156 cross-robot pairs. Green band 0.2 ≤ I ratio ≤ 3: {env_I['inside_works']}/{env_I['inside_n']} work", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_operating_envelope.png", dpi=150); plt.close(fig)

    summary = dict(works_drift_threshold_m=WORKS_DRIFT, envelope_inertia=env_I, envelope_mass=env_m,
                   by_inertia_ratio=binned(off, "inertia_ratio"), by_mass_ratio=binned(off, "mass_ratio"),
                   inertia_only_ablation_by_ratio=binned(abl_I, "inertia_ratio"),
                   fell_pairs=sorted([f"{r['src']}->{r['tgt']}" for r in off if r["fell"]]), n_pairs=len(off), overall=full, noise_floor_drift=diag,
                   spearman_vs_log_drift=corr, best_single_threshold=stumps, majority_baseline_accuracy=round(base_rate, 3),
                   decision_tree=rules, decision_tree_cv_accuracy=[round(float(v), 3) for v in cv],
                   ablation={k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in abl.items()},
                   sweep=sweep_tab, families=fam,
                   robots={r: {k: v for k, v in F[r].items() if k not in ("I", "M", "h")} |
                           dict(Ixx=F[r]["I"][0][0], Iyy=F[r]["I"][1][1], Izz=F[r]["I"][2][2]) for r in order})
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ["overall", "spearman_vs_log_drift", "best_single_threshold",
                                              "majority_baseline_accuracy", "decision_tree_cv_accuracy", "ablation", "families"]}, indent=1))
    print(rules)


if __name__ == "__main__":
    main()
