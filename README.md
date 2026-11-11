# SSR-ICLLL

Code and data accompanying the paper

> **Senses can be recovered from word vectors in static word embeddings**
> Kow Kuroda (Kyorin University Medical School), ICLLL 2026.

## Overview

A static word embedding assigns one vector to each word type, apparently collapsing a
polysemous word's senses into a single point. **SSR (Spectral Sense Recovery)** treats that
collapse as a *superposition*: it partitions a word's nearest neighbors spectrally and takes
each group's centroid as a sense vector. It uses only the geometry of the embedding (no sense
labels, no training), but the number of senses must be supplied.

The paper finds that SSR separates two superposed senses when they are comparably weighted,
but not when one dominates, and that the sense count cannot be read off the neighborhood. On
real words, scored against a null that keeps the clustering and breaks only its link to
meaning (the *swap null*), the neighbor clusters recover WordNet, French-WordNet and FrameNet
senses at a small but consistent margin.

**About the name.** SSR originally stood for *Simulated Self-Repair*: noisy copies of the word
vector were pulled toward the cluster centroids. That repair step leaves the centroids
essentially unchanged (cosine 0.998–0.999) and adds no accuracy (paper, Sec. III-D), so the
acronym is now read as Spectral Sense Recovery. The code still implements the repair step,
and the function and module names keep the original wording.

The method is implemented in the vendored `sense_explorer` package (v0.9.5 snapshot). The
upstream package lives at <https://github.com/kow-k/explore-senses-in-embeddings>. In this
snapshot only docstrings were changed (retracted accuracy figures and the DNA-repair analogy);
the code is unchanged.

## Contents

```
sense_explorer/     SSR implementation (frozen v0.9.5 snapshot; docstrings corrected)
io_utils.py         minimal embedding loader shared by the drivers
data/
  polysemous_300words.csv                     260 polysemous words x 8 semantic domains
  polysemous_300words-glove-wiki-{100,200,300}d.json   SSR sense vectors (Sec. IV-D, Fig. 2)
  bnc_verbs_120.csv                           115 BNC verbs x 8 semantic classes
  bnc_verbs_120_baseline.json                 one vector per verb (baseline)
  bnc_verbs_120_ssr.json                      SSR sense vectors (Sec. IV-E)
  wordnet_anchors.json                        WordNet-derived anchors
figures/
  fig2_crossword_dendrogram.png               Fig. 2
```

Scripts, by the part of the paper they reproduce:

| Paper | Script(s) | What it gives |
|---|---|---|
| III-C sense angle | `centroid_angles.py` | two-sense angles, 41–52° (unsupervised) |
| III-D what repair adds | `centroid_vs_ssr.py`, `noise_only_ssr.py` | repair vs. centroids alone; noise-only clustering |
| IV-A worked example, Table I | `sense_localization_glove.py` | *bank*, anchor-guided, 48.3° |
| IV-B WordNet reconstruction | `run_synset_mapping.py` (+ `synset_mapper.py`, `wordnet_anchors.py`) | 15/16 anchor-guided; 11/15 unsupervised |
| IV-C Table II (cross-lingual) | `table2_swap.py`, `run_table2_grid.sh` (+ `stage1_anchor.py`, `stage2_score_emb.py`) | lift vs. random partition and vs. swap null, full grid |
| IV-C FrameNet | `framenet_eval.py` | FrameNet 1.7 frames, swap null |
| IV-D cross-word coherence, Fig. 2 | `extract_senses_to_json.py`, `sense_cluster_radial.py`, `coherence_check.py` | Fig. 2; domain coherence and split coverage with controls |
| IV-E verb replication | `within_between_metrics.py`, `verb_ratio_check.py` | within/between ratio with controls |
| IV-F pseudo-words | `pseudo_word_test.py`, `adaptive_k.py`, `mono_vs_poly_k.py`, `deflation_test.py` | separation vs. dominance (Table III); sense-count rules; detection statistics |
| V future work (deflation) | `deflation_test.py` | minor-sense recovery after removing the dominant sense |

Helpers: `centroids_only.py` runs any driver with the repair step replaced by the centroids;
`rerun_centroids.sh` reruns III-C, IV-B, IV-D and IV-E that way. `compare_clusterings.py` draws
baseline-vs-SSR clusterings side by side. `stage2_eval.py` is the original Table II driver.

`framenet_anchors.py` is kept for the record only. It produced the FrameNet result of the
reviewed version, which is withdrawn: it does not read FrameNet, and its scoring overlaps its
anchors (paper, Appendix A).

## Installation

```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r) for r in ('wordnet', 'omw-1.4', 'framenet_v17')]"
```

## Embeddings (downloaded separately)

- **GloVe 6B** (Wikipedia + Gigaword), two distributions of the same vectors:
  - `glove.6B.{50,100,200,300}d.txt` from <https://nlp.stanford.edu/projects/glove/>, used by
    the evaluation drivers (`--embedding`);
  - `glove-wiki-gigaword-{100,200,300}.gz` via `gensim.downloader`, used by the
    `sense_explorer` drivers (`run_synset_mapping.py`, `extract_senses_to_json.py`,
    `centroid_angles.py`).
- **BNC skip-gram, 300d** (POS-tagged lemmas such as `run_VERB`), for Sec. IV-E only.

The French side of the cross-lingual test comes from WordNet/OMW (`lang='fra'`); no French
embedding is needed. The shell scripts read the GloVe files from `$EMB` (default
`./embeddings`) and the BNC model from `$BNC`.

## Reproduction

All commands run from the repository root. `E` is the directory holding the GloVe files.

**Sanity checks (no embeddings).**
```bash
python stage2_eval.py --selftest
python stage1_anchor.py --max-pairs 200
```

**III-C / III-D. Sense angle; what repair adds.**
```bash
python centroid_angles.py glove-wiki-gigaword-100.gz data/polysemous_300words.csv --example bank
python centroid_vs_ssr.py --embedding $E/glove.6B.100d.txt --topk 40,100   # also 200d, 300d
python noise_only_ssr.py  --embedding $E/glove.6B.100d.txt --topk 40       # also 300d
```

**IV-A / IV-B. Worked example and WordNet reconstruction.**
```bash
python sense_localization_glove.py --glove $E/glove.6B.100d.txt --words bank      # 48.3°
python run_synset_mapping.py --glove glove-wiki-gigaword-100.gz \
    --words bank bat cell crane mouse plant spring --method induce               # anchor-guided, 15/16
python run_synset_mapping.py --glove glove-wiki-gigaword-300.gz \
    --words bank bat cell crane mouse plant spring                               # unsupervised, 11/15
```

**IV-C. Table II and FrameNet.**
```bash
EMB=$E bash run_table2_grid.sh        # 50-300d x K = 30..100; all 32 cells and a summary grid
python framenet_eval.py --embedding $E/glove.6B.100d.txt --topk 40,100 --csv fn_100d.csv
```
`table2_swap.py` reports, for the same 1,500 pairs and clusters, the published scoring
(random-partition null) and the corrected scoring (target excluded from its synset centroids,
swap null). `framenet_eval.py` reports the swap-null lift with and without the frames' own
lexical units in the neighbor pool.

**IV-D. Cross-word coherence and Fig. 2.**
```bash
python extract_senses_to_json.py glove-wiki-gigaword-100.gz \
    data/polysemous_300words.csv data/polysemous_300words-glove-wiki-100d.json discover
python sense_cluster_radial.py data/polysemous_300words-glove-wiki-100d.json \
    -o figures/fig2_crossword_dendrogram.png \
    --exclude-defaults --no-title --no-labels --legend-min-size 15 \
    --figsize 12 --branch-linewidth 1.4
python coherence_check.py --embedding $E/glove.6B.100d.txt \
    --senses data/polysemous_300words-glove-wiki-100d.json --words data/polysemous_300words.csv
```

**IV-E. Verb replication.**
```bash
python within_between_metrics.py data/bnc_verbs_120_baseline.json data/bnc_verbs_120_ssr.json
python verb_ratio_check.py --embedding /path/to/bnc/model.txt \
    --verbs data/bnc_verbs_120.csv --senses data/bnc_verbs_120_ssr.json
```

**IV-F. Pseudo-words and the sense count.**
```bash
python pseudo_word_test.py --embedding $E/glove.6B.100d.txt --pairs 200 --topk 50,100 --kstats   # Table III
python mono_vs_poly_k.py   --embedding $E/glove.6B.100d.txt --topk 40,100
python deflation_test.py   --embedding $E/glove.6B.100d.txt --pairs 200 --topk 50 --csv defl_100d.csv
```
`deflation_test.py` also gives the detection statistics of Sec. IV-F (written to
`*_detect.csv`) and, with `--show-words bank,bat,spring,crane,mouse,plant,cell --rounds 3`,
iterative deflation on real words. Repeat the runs at 300d for the second column of Table III.

### Changes from the reviewed version

The paper's Appendix A lists how the evaluations differ from the reviewed version. In brief:
Table II's random-partition null was lenient, and the swap null replaces it; the WordNet
reconstruction is anchor-guided; the FrameNet result of `framenet_anchors.py` is withdrawn and
replaced by `framenet_eval.py`; IV-D and IV-E gain controls; and the pseudo-word test is new.

## Citation

```bibtex
@inproceedings{kuroda2026ssr,
  title     = {Senses can be recovered from word vectors in static word embeddings},
  author    = {Kuroda, Kow},
  booktitle = {Proc. ICLLL 2026},
  year      = {2026}
}
```

## Acknowledgment

This work was developed in close collaboration with Claude (Anthropic), which contributed to
implementation, empirical evaluation, and manuscript preparation.

## License

MIT — see [LICENSE](LICENSE).
