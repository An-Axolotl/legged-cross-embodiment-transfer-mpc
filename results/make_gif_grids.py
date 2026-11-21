"""Print the README GIF grids for every transfer (original recording and rich profile)."""
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
COLS = 5
NAMES = {"go1": "Go1", "go2": "Go2", "aliengo": "Aliengo", "mini_cheetah": "Mini Cheetah", "b2": "B2", "hyqreal": "HyQReal",
         "a1": "A1", "laikago": "Laikago", "solo12": "Solo-12", "anymal_b": "ANYmal B", "anymal_c": "ANYmal C", "b1": "B1",
         "hyq": "HyQ", "quadDoggo": "quadDoggo (Go1 clone)", "go1_perturbed": "Go1, masses ±20 %, base ×1.3"}
A = json.load(open(f"{D}/analysis/summary.json"))
order = list(A["robots"])  # sorted by mass


def grid(cells):
    out = []
    for i in range(0, len(cells), COLS):
        row = cells[i:i + COLS]
        row += [("", "")] * (COLS - len(row))
        if i == 0:
            out.append("| " + " | ".join(c[0] for c in row) + " |")
            out.append("|" + "---|" * COLS)
        else:
            out.append("| " + " | ".join(f"**{c[0]}**" if c[0] else "" for c in row) + " |")
        out.append("| " + " | ".join(c[1] for c in row) + " |")
    return "\n".join(out)


cells = []
for t in ["go1", "quadDoggo", "go1_perturbed"] + [r for r in order if r != "go1"]:
    r = json.load(open(f"{D}/original/original_go1_to_{t}.json"))
    res = f"fell at {r['fell_at_s']:.1f} s" if r["fell"] else f"xy err {r['mean_err_x_m'] + r['mean_err_y_m']:.2f} m"
    cells.append((f"Go1 → {NAMES[t]}", f'<img src="media/gifs/original_go1_to_{t}.gif" width="180"><br><sub>{res}</sub>'))
print("<!-- original -->\n" + grid(cells))

rows = {r["tgt"]: r for r in json.load(open(f"{D}/analysis/matrix_rows.json")) if r["src"] == "go1"}
cells = []
for t in order:
    r = rows[t]
    res = "**fell**" if r["fell"] else f"drift {r['drift']:.2f} m ({r['label']})"
    cells.append((f"Go1 → {NAMES[t]}", f'<img src="media/gifs/rich_go1_to_{t}.gif" width="180"><br><sub>{res}</sub>'))
print("<!-- rich -->\n" + grid(cells))
