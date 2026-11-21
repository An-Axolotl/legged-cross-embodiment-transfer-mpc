"""Runs every Part-B experiment in parallel (each worker gets its own acados code-generation dir).

    source deps/env.sh
    python analysis/run_experiments.py --workers 24          # ~25 min on 32 cores
    python analysis/run_experiments.py --only native,matrix  # subsets

Restartable: a job whose result file exists is skipped. Results: results/<experiment>/<tag>.json
(+ <tag>_traj.npz for transfers), videos: media/videos/.

Experiments
  native    each robot under its own NMPC on the 20 s "rich" command profile (also the transfer sources)
  matrix    every robot's recorded model -> every robot  (13 x 13, all components transferred)
  ablation  same pairs, but only ONE transferred component (inertia | legM | bias) besides the references
  sweep     Go1 -> Go1 with one source quantity scaled by k in {0.1 ... 16}, 3 seeds
  original  the Jan-2025 Go1 recording (40 s, human-driven) -> every robot, with videos
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIM = f"{ROOT}/mpc_transfer/Quadruped-PyMPC/simulation/transfer.py"
RES, VID = f"{ROOT}/results", f"{ROOT}/media/videos"
ROBOTS = ["go1", "go2", "aliengo", "mini_cheetah", "b2", "hyqreal",
          "a1", "laikago", "solo12", "anymal_b", "anymal_c", "b1", "hyq"]
NOISE = ["--noise-frac", "0.044"]      # = 2 N·m on Go1 (original experiment), proportional elsewhere
RICH_STEPS = "10000"                    # 20 s at 2 ms


def jobs(only):
    J = []
    if "native" in only:
        for r in ROBOTS:
            J.append(("native", r, f"native_{r}", ["record", "--profile", "rich", "--steps", RICH_STEPS,
                      "--out", f"{RES}/native/native_{r}.npz", "--video", f"{VID}/native_{r}.mp4", "--seed", "0"]))
    if "matrix" in only:
        for s in ROBOTS:
            for t in ROBOTS:
                vid = ["--video", f"{VID}/rich_{s}_to_{t}.mp4"] if s == "go1" else []
                J.append(("matrix", t, f"{s}_to_{t}", ["transfer", "--src", f"{RES}/native/native_{s}.npz",
                          "--steps", RICH_STEPS, "--seed", "1"] + vid))
    if "ablation" in only:
        for part in ["ref", "inertia", "legM", "bias"]:
            parts = "ref" if part == "ref" else f"ref,{part}"
            for s in ROBOTS:
                for t in ROBOTS:
                    if s != t:
                        J.append(("ablation", t, f"{part}__{s}_to_{t}", ["transfer", "--src", f"{RES}/native/native_{s}.npz",
                                  "--steps", RICH_STEPS, "--seed", "1", "--parts", parts]))
    if "sweep" in only:
        for q in ["inertia", "legM", "bias"]:
            for k in ["0.1", "0.25", "0.5", "1", "2", "4", "8", "16"]:
                for seed in ["1", "2", "3"]:
                    J.append(("sweep", "go1", f"{q}_x{k}_s{seed}", ["transfer", "--src", f"{RES}/native/native_go1.npz",
                              "--steps", RICH_STEPS, "--seed", seed, f"--scale-{q}", k]))
    if "original" in only:
        src = f"{ROOT}/data/go1_original_2025-01-16.npz"
        for t in ROBOTS + ["quadDoggo"]:
            J.append(("original", t, f"original_go1_to_{t}", ["transfer", "--src", src, "--steps", "20000", "--seed", "1",
                      "--video", f"{VID}/original_go1_to_{t}.mp4"]))
        J.append(("original", "go1", "original_go1_to_go1_perturbed", ["transfer", "--src", src, "--steps", "20000",
                  "--seed", "1", "--mass-scale", "1.3", "--video", f"{VID}/original_go1_to_go1_perturbed.mp4"]))
    return J


def work(job):
    exp, robot, tag, args = job
    out = f"{RES}/{exp}/{tag}.json"
    if os.path.exists(out):
        return tag, "cached"
    if args[0] == "transfer":
        args = args + ["--out", out]
    slot = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    env = dict(os.environ, XFER_ROBOT=robot, XFER_CODEGEN_DIR=f"/tmp/xfer_codegen/w{slot}_{robot}")
    wd = f"/tmp/xfer_codegen/w{slot}_{robot}_cwd"
    os.makedirs(wd, exist_ok=True)
    p = subprocess.run(["python", SIM] + args + NOISE, env=env, cwd=wd, capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        err = (p.stderr or p.stdout).strip().splitlines()[-3:]
        return tag, "ERROR " + " | ".join(err)
    r = json.loads(line[-1][7:])
    r.update(experiment=exp, tag=tag)
    json.dump(r, open(out, "w"))
    return tag, "fell" if r["fell"] else "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() * 3 // 4))
    ap.add_argument("--only", default="native,matrix,ablation,sweep,original")
    a = ap.parse_args()
    only = a.only.split(",")
    for d in ["native", "matrix", "ablation", "sweep", "original"]:
        os.makedirs(f"{RES}/{d}", exist_ok=True)
    os.makedirs(VID, exist_ok=True)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for phase in (["native"] if "native" in only else []) + [[x for x in only if x != "native"]]:
            J = jobs(phase)
            for i, (tag, status) in enumerate(pool.imap_unordered(work, J, chunksize=1)):
                if status != "cached":
                    print(f"[{time.time()-t0:6.0f}s] {i+1}/{len(J)} {tag}: {status}", flush=True)
    print("ALL DONE")
