#!/usr/bin/env bash
# run_table2_grid.sh -- Table II with the swap null over the full grid:
#   dimensions 50/100/200/300 x K = 30..100 (step 10), all cells reported.
# Run from the repository root:
#   bash run_table2_grid.sh                                  # plain pool (Table II)
#   TAG=dedup90 POOLARGS="--pool dedup --dedup-q 0.9" bash run_table2_grid.sh
#   TAG=mmr70   POOLARGS="--pool mmr --mmr-lambda 0.7" bash run_table2_grid.sh
#   DIMS="" TAG=dedup90 bash run_table2_grid.sh               # only reprint the grid
# Outputs: t2grid[_TAG]_<d>d.csv / .log
set -euo pipefail
EMB=${EMB:-embeddings}   # directory holding the GloVe files
KS=${KS:-30,40,50,60,70,80,90,100}
TAG=${TAG:-}
PFX="t2grid${TAG:+_$TAG}"
for d in ${DIMS-50 100 200 300}; do
  echo "=== ${d}d ${TAG:+($TAG)} ==="
  python table2_swap.py --embedding "$EMB/glove.6B.${d}d.txt" --topk "$KS" ${POOLARGS:-} \
      --csv "${PFX}_${d}d.csv" | tee "${PFX}_${d}d.log" | sed -n '/=== summary/,$p'
done
PFX="$PFX" python - <<'PY'
import csv, glob, os, re, numpy as np
pfx = os.environ["PFX"]
rows = []
for f in sorted(glob.glob(f"{pfx}_*d.csv")):
    m = re.search(r"_(\d+)d\.csv$", f)
    if not m or not re.fullmatch(re.escape(pfx) + r"_\d+d\.csv", os.path.basename(f)):
        continue
    rows += [dict(r, dim=int(m.group(1))) for r in csv.DictReader(open(f))]
if not rows:
    raise SystemExit("no grid files found for " + pfx)
print(f"\nswap-null lift (target excluded), by dimension x K   [{pfx}]")
Ks = sorted({int(r["K"]) for r in rows}); dims = sorted({r["dim"] for r in rows})
print("  dim   " + " ".join(f"{k:>7d}" for k in Ks) + "    mean")
for d in dims:
    cells = []
    for k in Ks:
        R = [r for r in rows if r["dim"] == d and int(r["K"]) == k and r["swap"] in ("True", "False")]
        a = np.mean([r["ssr_x"] == "True" for r in R]); c = np.mean([r["swap"] == "True" for r in R])
        cells.append(a - c)
    print(f"  {d:3d}d  " + " ".join(f"{v:+7.3f}" for v in cells) + f"  {np.mean(cells):+7.3f}")
PY
