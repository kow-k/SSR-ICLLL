# SSR-ICLLL

Code and data accompanying the paper

> **Senses can be recovered from word vectors in static word embeddings via Simulated Self-Repair**
> Kow Kuroda (Kyorin University School of Medicine), ICLLL 2026.

## Overview

A static word embedding assigns one vector to each word type, apparently collapsing a
polysemous word's senses into a single point. **Simulated Self-Repair (SSR)** treats that
collapse as a *superposition* rather than a destruction: noisy copies of a word's vector are
allowed to settle into attractor basins defined by spectral clustering of the word's
neighbors, and the per-basin means are recovered as sense vectors. The procedure uses only
the geometry of the embedding — no sense inventory, no training, no supervision.

The method is implemented in the vendored `sense_explorer` package (v0.9.5). The upstream,
actively developed package lives at
<https://github.com/kow-k/explore-senses-in-embeddings>; a frozen snapshot is included here
so this repository is self-contained.

## Contents

```
sense_explorer/     SSR implementation (frozen v0.9.5 snapshot)
io_utils.py         minimal embedding loader shared by the validation drivers
data/               experiment inputs and SSR outputs used in the paper
  polysemous_300words.csv                     260 polysemous words x 8 semantic domains
  polysemous_300words-glove-wiki-{100,200,300}d.json   SSR sense vectors (cross-word study)
  bnc_verbs_120.csv                           115 BNC verbs x 8 semantic classes
  bnc_verbs_120_baseline.json                 one vector per verb (baseline)
  bnc_verbs_120_ssr.json                      SSR-separated sense vectors (verb study)
  wordnet_anchors.json                        WordNet-derived anchors
figures/
  fig2_crossword_dendrogram.png               rendered Fig. 2 (cross-word coherence)

Driver scripts, grouped by the validation they reproduce:
  run_synset_mapping.py  synset_mapper.py          WordNet / FrameNet reconstruction (SIV-B)
  wordnet_anchors.py     framenet_anchors.py
  stage1_anchor.py  stage2_score_emb.py  stage2_eval.py   cross-lingual anchor recovery (SIV-C)
  extract_senses_to_json.py  extract_baseline_to_json.py  cross-word coherence (SIV-D)
  sense_cluster_radial.py  compare_clusterings.py  within_between_metrics.py
  sense_localization_glove.py                       sense geometry / localization (SIII)
```

## Installation

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Embeddings (downloaded separately)

The large embedding files are not bundled. Download as needed:

- **GloVe** (Wikipedia+Gigaword, 50/100/200/300d) — via `gensim.downloader`
  (`glove-wiki-gigaword-{50,100,200,300}`) or <https://nlp.stanford.edu/projects/glove/>.
- **FastText** `cc.en.300` (cross-lingual validation) —
  <https://fasttext.cc/docs/en/crawl-vectors.html>. The cross-lingual driver
  scores on this single English embedding; the French side of each pair is
  supplied lexically through WordNet/OMW (`lang='fra'`), so no French embedding
  is required.
- **Google News word2vec 300d** (register comparison) — standard distribution.

## Reproduction

The paper reports four externally validated results. The frozen method
(`sense_explorer/`), the shared loader (`io_utils.py`) and the experiment data
(`data/`) are all included here. Toy/self-tests need no embeddings; full runs
need the embedding files above. All commands are run from the repository root.

**0. Sanity checks (no embeddings).**
```bash
python stage2_eval.py --selftest          # Stage-2 scoring + McNemar on synthetic data
python stage1_anchor.py --max-pairs 200   # Stage-1 cross-lingual anchor gate (WordNet/OMW)
```

**1. WordNet / FrameNet reconstruction (Table I, SIV-B).**
Separate senses for the test words, then assign each sense vector to its nearest
WordNet synset / FrameNet frame by neighbor-coverage.
```bash
python run_synset_mapping.py --glove glove-wiki-gigaword-300.gz --preset all
```
`wordnet_anchors.py` and `framenet_anchors.py` build the anchor sets
(FrameNet via NLTK).

**2. Cross-lingual anchor recovery (Table II, SIV-C).**
For English-French translation pairs, fix the shared gold synset through the
Inter-Lingual Index, fix *k* to the known sense count, and test whether SSR
assigns the cluster containing the shared sense above a matched-null baseline
(McNemar's exact test, with a bootstrap 95% CI and a top-*k* sweep). The driver
scores on a single English embedding; the French side is supplied lexically
through WordNet/OMW.
```bash
python stage1_anchor.py                                       # Stage 1: clean, separable anchors M
python stage2_eval.py --embedding cc.en.300.vec.gz --topk 40,60,100
```

**3. Cross-word semantic coherence (Fig. 2, SIV-D).**
Export SSR sense vectors (and the no-SSR baseline) to JSON, then pool and cluster
them with no source-word labels; they group by semantic domain. The paper's
`data/polysemous_300words-glove-wiki-*.json` were produced by:
```bash
python extract_senses_to_json.py  glove-wiki-gigaword-100.gz \
    data/polysemous_300words.csv data/polysemous_300words-glove-wiki-100d.json discover
python extract_baseline_to_json.py glove-wiki-gigaword-100.gz \
    data/polysemous_300words.csv data/polysemous_300words-baseline-100d.json
```
Render Fig. 2 (the bundled `figures/fig2_crossword_dendrogram.png`):
```bash
python sense_cluster_radial.py data/polysemous_300words-glove-wiki-100d.json \
    -o figures/fig2_crossword_dendrogram.pdf \
    --exclude-defaults --no-title --no-labels --legend-min-size 15 \
    --figsize 12 --branch-linewidth 1.4
```
`compare_clusterings.py` draws baseline-vs-SSR clusterings side by side.

**4. BNC verb replication (SIV-E).**
On POS-tagged, lemmatized BNC embeddings SSR produces local sense-resolution
refinement, quantified as the within-word vs. between-word distance ratio from
the bundled verb JSONs:
```bash
python within_between_metrics.py data/bnc_verbs_120_baseline.json data/bnc_verbs_120_ssr.json
```

**Sense geometry (SIII).** `sense_localization_glove.py` reproduces the
inter-sense angle and mixing-coefficient analysis on GloVe:
```bash
python sense_localization_glove.py --glove glove.6B.100d.txt --use-sense-explorer
```

### A note on scope
SSR decides sense **count** by spectral clustering of each word's neighbor graph
(`sense_explorer/spectral.py`). The scripts here reproduce the four validations
reported in the paper; superseded count-accuracy instrumentation is deliberately
**not** included.

## Citation

```bibtex
@inproceedings{kuroda2026ssr,
  title     = {Senses can be recovered from word vectors in static word embeddings
               via Simulated Self-Repair},
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
