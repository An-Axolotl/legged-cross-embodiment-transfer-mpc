"""Print the Markdown result tables used in the README."""
import glob
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(f"{D}/analysis/summary.json")) if os.path.exists(f"{D}/analysis/summary.json") else None

print("#### Part A: jump (1 m forward, 0.5 m CoM rise)\n")
print("| run | formulation | converged (iters) | mass [kg] | final base x [m] | peak abs τ [N·m] | Σ abs τ [N·m] | max abs τ / limit | pre-launch CoM margin [m] |")
print("|---|---|---|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{D}/jump/*.json")):
    r = json.load(open(f)); tag = os.path.basename(f)[:-5]
    print(f"| `{tag}` | {r['formulation']} | {r['converged']} ({r['iterations']}) | {r['mass_kg']:.2f} | {r['final_base_xyz'][0]:.3f} | "
          f"{r['peak_abs_tau_Nm']:.0f} | {r['total_abs_tau_Nm']:.0f} | {r['max_tau_over_limit']:.2f} | {r['prelaunch_com_margin_m']:+.3f} |")

if A:
    print("\n#### Part B: the 13 robots, each under its own NMPC (20 s rich command profile)\n")
    print("| robot | family | mass [kg] | controller mass [kg] | hip height [m] | torque limit [N·m] | knees (front/rear) | I_xx, I_yy, I_zz [kg·m²] | own run | video |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r, f in A["robots"].items():
        nat = json.load(open(f"{D}/native/native_{r}.json")) if os.path.exists(f"{D}/native/native_{r}.json") else None
        own = "–" if nat is None else ("fell at %.1f s" % nat["fell_at_s"] if nat["fell"] else "completes")
        flag = " ⚠" if abs(f["mass_cfg"] / f["mass_model"] - 1) > 0.05 else ""
        print(f"| {r} | {f['family']} | {f['mass_model']:.1f} | {f['mass_cfg']:.1f}{flag} | {f['hip_height']:.2f} | {f['tau_max']:.0f} | "
              f"{f['knee_front']}/{f['knee_rear']} | {f['Ixx']:.2f}, {f['Iyy']:.2f}, {f['Izz']:.2f} | {own} | [mp4](media/videos/native_{r}.mp4) |")

print("\n#### Part B: the original Jan-2025 Go1 recording (human-driven, 40 s) replayed on every robot\n")
print("| target | survived | mean abs x err [m] | mean abs y err [m] | mean abs z err [m] | Σ error (x+y+z) | video |")
print("|---|---|---|---|---|---|---|")
rows = [json.load(open(f)) for f in glob.glob(f"{D}/original/*.json")]
for r in sorted(rows, key=lambda r: r["err_total"]):
    name = r["tag"].replace("original_go1_to_", "")
    surv = "fell at %.1f s" % r["fell_at_s"] if r["fell"] else "yes (40 s)"
    print(f"| {name} | {surv} | {r['mean_err_x_m']:.3f} | {r['mean_err_y_m']:.3f} | {r['mean_err_z_m']:.3f} | {r['err_total']:.0f} | "
          f"[mp4](media/videos/{r['tag']}.mp4) |")
