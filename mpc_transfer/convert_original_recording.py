"""Convert the original Jan-2025 Go1 recording (ref_base_velocities.pkl written by
simulation_original_jan2025.py with is_transfer=False) into the .npz format used by
simulation/transfer.py.

    python convert_original_recording.py ref_base_velocities.pkl ../data/go1_original_2025-01-16.npz
"""
import pickle
import sys

import numpy as np

LEGS = ["FL", "FR", "RL", "RR"]
src, dst = sys.argv[1], sys.argv[2]
ref_lin, ref_ang, qfrc, mass, inertia, x, y, z = pickle.load(open(src, "rb"))
np.savez_compressed(dst,
                    ref_lin=np.array(ref_lin), ref_ang=np.array(ref_ang),
                    qfrc=np.array([[np.asarray(h[l]) for l in LEGS] for h in qfrc]),
                    mass=np.array([[np.asarray(M[l]) for l in LEGS] for M in mass]),
                    inertia=np.array(inertia),
                    base_pos=np.stack([x, y, z], 1))
print("wrote", dst, len(ref_lin), "steps")
