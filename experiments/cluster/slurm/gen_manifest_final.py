"""
Final head-to-head: current GPry defaults vs the proposed configuration.

Both arms are re-run under IDENTICAL exclusive-node allocation. The earlier
125-run study shared nodes (four jobs at once on one node), which made per-eval
cost swing 45->107 ms at identical n and d, so its wall-clock numbers cannot be
compared against fresh runs.

  BASE : S0, length-scale ceiling 1e2, SciPy default ftol   (today's defaults)
  PROP : S2, length-scale ceiling 1e3, ftol 1e-5            (proposed)

Fields: kind d arm seed ftol lsmax
"""
CASES = [("gauss", 30), ("gauss", 16), ("gauss", 8),
         ("curved", 5), ("multimode", 5)]
SEEDS = [1, 2, 3, 4, 5]
ARMS = [("BASE", "S0", "0", "0"), ("PROP", "S2", "1e-5", "1000")]

lines = []
for kind, d in CASES:                      # d=30 first: longest, fills the tail
    for label, arm, ftol, lsmax in ARMS:
        for seed in SEEDS:
            lines.append(f"{kind} {d} {arm} {seed} {ftol} {lsmax}")
import os  # noqa: E402
# Write the manifest NEXT TO THIS SCRIPT, not into the caller's cwd: the
# sbatch files read it by a path relative to experiments/cluster/.
_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest_final.txt")
with open(_OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"{len(lines)} tasks -> manifest_final.txt")
for kind, d in CASES:
    print(f"  {kind:10s} d={d:<3d} 2 arms x {len(SEEDS)} seeds")
