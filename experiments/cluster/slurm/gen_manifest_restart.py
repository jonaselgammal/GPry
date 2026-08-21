"""Generate the restart-efficiency study manifest (one line = one cluster task)."""
ARMS = ["S0", "S1", "S2", "S3", "S4"]
SEEDS = [1, 2, 3, 4, 5]
# d=5 is deliberately NON-Gaussian (curved + 4-mode): the study has to show that
# a cheaper fit budget keeps robustness, not just that it is faster on easy
# unimodal Gaussians. d=8/16/30 give the speed/scaling curve.
CASES = [("curved", 5), ("multimode", 5), ("gauss", 8), ("gauss", 16), ("gauss", 30)]

lines = []
# Order matters: SLURM starts array tasks roughly in order and d=30 is by far
# the longest (~3.7 h at S0), so put it FIRST to fill the wall-clock tail.
for kind, d in sorted(CASES, key=lambda c: -c[1]):
    for arm in ARMS:
        for seed in SEEDS:
            lines.append(f"{kind} {d} {arm} {seed}")

import os  # noqa: E402
# Write the manifest NEXT TO THIS SCRIPT, not into the caller's cwd: the
# sbatch files read it by a path relative to experiments/cluster/.
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_restart.txt")
with open(_OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"{len(lines)} tasks -> manifest_restart.txt")
for kind, d in sorted(CASES, key=lambda c: -c[1]):
    print(f"  {kind:10s} d={d:<3d} {len(ARMS)*len(SEEDS)} runs")
