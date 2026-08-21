"""Manifest for the POST-PR#4 acquisition-sampler head-to-head.

Every NUTS-vs-nested comparison this project owns is PRE-PR#4, when the
`length_scale_prior` bug made every fit optimise over 10 decades instead of 4.
That bug plausibly hurt NUTS most (its d=30 loop was ~85% fit), so the existing
d=30 result -- UltraNest 1840 s / KL 0.0447 vs NUTS 11822 s / KL 0.0475 -- may be
an artefact of the bug rather than a property of the samplers. This campaign
re-runs the comparison on post-fix code, at PROP defaults, on exclusive nodes.

Both arms are identical except for the acquisition sampler; the final MC is
matched to it in `run_restart_study.py`.

Longest-first so the wall-clock tail is filled early: UltraNest is the unknown,
and within a sampler cost grows with d.
"""
CASES = [("gauss", 30), ("gauss", 16), ("multimode", 5), ("curved", 5),
         ("gauss", 8)]
SAMPLERS = ["ultranest", "nuts"]      # slowest arm first
SEEDS = [1, 2, 3, 4, 5]
ARM = "S2"                            # PROP: informed starts only

lines = [f"{k} {d} {ARM} {s} {smp}"
         for smp in SAMPLERS for k, d in CASES for s in SEEDS]
import os  # noqa: E402
# Write the manifest NEXT TO THIS SCRIPT, not into the caller's cwd: the
# sbatch files read it by a path relative to experiments/cluster/.
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_h2h.txt")
with open(_OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"{len(lines)} tasks -> manifest_h2h.txt")
for ln in lines[:3] + ["..."] + lines[-3:]:
    print(" ", ln)
