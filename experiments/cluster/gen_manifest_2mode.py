"""
Generate manifest_2mode.txt for the Tier-B separation sweep.

Grid: d x sep x sampler x seed.  Ordered SEED-MAJOR so that
  --array=1-30   runs a complete seed-1 sweep (a self-contained pilot:
                 both samplers, all separations, all dimensions),
  --array=1-90   runs all 3 seeds.

Columns per line:  <d> <sep> <wr> <max_total> <ckpts> <sampler> <seed>
"""
DS = [2, 8, 16]
SEPS = [2, 4, 6, 8, 10]
SAMPLERS = ["nuts", "ultranest"]
SEEDS = [1, 2, 3]
WR = 1.0                                  # Tier B = equal weights
BUDGET = {2: 300, 8: 600, 16: 1000}
CKPTS = {2: "200,300", 8: "400,600", 16: "700,1000"}

lines = []
for seed in SEEDS:
    for d in DS:
        for sep in SEPS:
            for sampler in SAMPLERS:
                lines.append(f"{d} {sep} {WR:g} {BUDGET[d]} {CKPTS[d]} {sampler} {seed}")

with open("manifest_2mode.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote manifest_2mode.txt with {len(lines)} lines "
      f"({len(SEEDS)} seeds x {len(DS)} dims x {len(SEPS)} seps x {len(SAMPLERS)} samplers)")
print(f"pilot (seed 1): --array=1-{len(DS)*len(SEPS)*len(SAMPLERS)}   full: --array=1-{len(lines)}")
