#!/usr/bin/env bash
# rerun_centroids.sh -- rerun the validations of Sections III-C, IV-A, IV-B, IV-D, IV-E
# with Spectral Sense Recovery (cluster centroids, no repair step), next to the
# original repair runs, so every number in the paper can be checked or replaced.
# (Section IV-C / Table II already measures the centroids; see centroid_vs_ssr.py.)
#
# Run from the repository root:
#   bash rerun_centroids.sh                 # all steps
#   bash rerun_centroids.sh angles synset   # only some: angles synset coherence verbs
# Paths can be overridden:
#   EMB=/path/to/glove BNC=/path/to/bnc/model.txt bash rerun_centroids.sh
set -euo pipefail

EMB=${EMB:-embeddings}   # directory holding the GloVe files
BNC=${BNC:-embeddings/bnc-skipgram/model.txt}
OUT=rerun_centroids
mkdir -p "$OUT"
STEPS=${*:-angles synset coherence verbs}
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
log() { echo; echo "=== $* ==="; }

# --- III-C sense angle + IV-A worked example (bank), unsupervised, k=2 -------------
if has angles; then
  for d in 100 200 300; do
    log "angles ${d}d"
    python centroid_angles.py "$EMB/glove-wiki-gigaword-$d.gz" data/polysemous_300words.csv \
        --example bank | tee "$OUT/angles_${d}d.txt"
  done
fi

# --- IV-B WordNet reconstruction: original repair vs centroids ---------------------
if has synset; then
  for mode in repair centroid; do
    log "synset mapping ($mode)"
    if [ $mode = repair ]; then RUN="python"; else RUN="python centroids_only.py"; fi
    $RUN run_synset_mapping.py --glove "$EMB/glove-wiki-gigaword-300.gz" --words bank bat cell crane mouse plant spring \
        > "$OUT/synset_$mode.txt" 2>&1 || echo "  (failed; see $OUT/synset_$mode.txt)"
    grep -A40 "OVERALL SUMMARY" "$OUT/synset_$mode.txt" | head -45 || true
  done
fi

# --- IV-D cross-word coherence: centroid sense vectors, Fig. 2 -----------------------
if has coherence; then
  for d in 100 200 300; do
    log "coherence extract ${d}d (centroids)"
    python centroids_only.py extract_senses_to_json.py "$EMB/glove-wiki-gigaword-$d.gz" \
        data/polysemous_300words.csv "$OUT/polysemous_300words-centroids-${d}d.json" discover \
        > "$OUT/extract_${d}d.log" 2>&1
    python - "$OUT/polysemous_300words-centroids-${d}d.json" \
             "data/polysemous_300words-glove-wiki-${d}d.json" <<'PY'
import json, sys
for p in sys.argv[1:]:
    d = json.load(open(p)); print(f"  {p}: {len(d)} words, {sum(len(v) for v in d.values())} sense vectors")
PY
  done
  log "Fig. 2 from centroid vectors (100d)"
  python sense_cluster_radial.py "$OUT/polysemous_300words-centroids-100d.json" \
      -o "$OUT/fig2_crossword_dendrogram_centroids.pdf" \
      --exclude-defaults --no-title --no-labels --legend-min-size 15 \
      --figsize 12 --branch-linewidth 1.4 > "$OUT/fig2.log" 2>&1 \
      && echo "  wrote $OUT/fig2_crossword_dendrogram_centroids.pdf" \
      || echo "  (render failed; see $OUT/fig2.log)"
fi

# --- IV-E BNC verb replication: within/between ratio --------------------------------
if has verbs; then
  log "BNC verbs extract (centroids)"
  python centroids_only.py extract_senses_to_json.py "$BNC" data/bnc_verbs_120.csv \
      "$OUT/bnc_verbs_120_centroids.json" discover > "$OUT/extract_bnc.log" 2>&1 \
      || echo "  (failed; see $OUT/extract_bnc.log)"
  log "within/between: baseline vs original repair"
  python within_between_metrics.py data/bnc_verbs_120_baseline.json data/bnc_verbs_120_ssr.json \
      | tee "$OUT/verbs_repair.txt" | tail -25
  log "within/between: baseline vs centroids"
  python within_between_metrics.py data/bnc_verbs_120_baseline.json "$OUT/bnc_verbs_120_centroids.json" \
      | tee "$OUT/verbs_centroid.txt" | tail -25
fi

echo; echo "all outputs in $OUT/  (send me the folder as a .zip)"
