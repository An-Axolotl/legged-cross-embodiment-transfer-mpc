"""Summary figures for Part A (run after run_jumps.sh)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "../results/jump"
dt = 0.01

# 1) same task, same mass / limbs / torque limit: quadruped vs humanoid torque profiles
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, tag, title in [(axs[0], "anymal_same", "Quadruped (ANYmal morphology)"),
                       (axs[1], "talos_same", "Humanoid (Talos legs morphology)")]:
    tau = np.load(f"{D}/{tag}_tau.npz", allow_pickle=True)["tau"]
    r = json.load(open(f"{D}/{tag}.json"))
    t = np.arange(len(tau)) * dt
    ax.plot(t, tau, lw=1)
    ax.axhline(80, ls="--", c="k", lw=0.8); ax.axhline(-80, ls="--", c="k", lw=0.8)
    ax.set_title(f"{title}\npeak {r['peak_abs_tau_Nm']:.0f} N·m, Σ|τ| {r['total_abs_tau_Nm']:.0f} N·m", fontsize=10)
    ax.set_xlabel("time [s]"); ax.grid(alpha=0.3)
axs[0].set_ylabel("joint torque [N·m]  (dashed: 80 N·m limit)")
fig.suptitle("1 m long, 0.5 m high jump, both robots 9.07 kg (20 lb), 38 cm thigh, 33 cm shank", fontsize=10)
fig.tight_layout(); fig.savefig(f"{D}/fig_quadruped_vs_humanoid_torques.png", dpi=140); plt.close(fig)

# 2) total effort / peak torque bar chart across embodiments
rows = [json.load(open(f"{D}/{f}")) for f in sorted(os.listdir(D)) if f.endswith(".json")]
rows = [r for r in rows if r["formulation"] == "inverse"]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for same, ax in [(False, axs[0]), (True, axs[1])]:
    sub = [r for r in rows if r["same"] == same]
    names = [r["robot"] for r in sub]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, [r["peak_abs_tau_Nm"] for r in sub], 0.4, label="peak |τ| [N·m]")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [r["total_abs_tau_Nm"] for r in sub], 0.4, color="tab:orange", label="Σ|τ| [N·m]")
    ax.set_xticks(x, names); ax.set_ylabel("peak |τ| [N·m]"); ax2.set_ylabel("Σ|τ| over trajectory [N·m]")
    ax.set_title("stock robots" if not same else "same mass / limb lengths / torque limit", fontsize=10)
    ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(f"{D}/fig_effort_by_robot.png", dpi=140); plt.close(fig)
print("figures written to", D)
