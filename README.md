# SSR-iclll

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
data/               experiment inputs and SSR outputs used in the paper
  polysemous_300words.csv                     260 polysemous words x 8 semantic domains
  polysemous_300words-glove-wiki-{100,200,300}d.json   SSR sense vectors (cross-word study)
  bnc_verbs_120.csv                           115 BNC verbs x 8 semantic classes
  bnc_verbs_120_baseline.json                 one vector per verb (baseline)
  bnc_verbs_120_ssr.json                      SSR-separated sense vectors (verb study)
  wordnet_anchors.json                        WordNet-derived anchors
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
- **FastText** `cc.en.300`, `cc.fr.300` (cross-lingual validation) —
  <https://fasttext.cc/docs/en/crawl-vectors.html>.
- **Google News word2vec 300d** (register comparison) — standard distribution.

## Reproduction

The paper reports four externally validated results. The frozen method and the experiment
data are included here; the experiment **driver scripts are being added** (they are being
de-tangled from unrelated, superseded instrument code before publication). Each experiment:

1. **WordNet / FrameNet reconstruction (Table, §IV-B).** Separate senses for the test words,
   then assign each sense vector to its nearest WordNet synset / FrameNet frame by
   neighbor-coverage. Inputs: GloVe + `data/wordnet_anchors.json`.
2. **Cross-lingual anchor recovery (Table II, §IV-C).** For English–French translation pairs,
   fix the shared gold synset via the Inter-Lingual Index, fix *k* to the known sense count,
   and test whether SSR assigns the cluster containing the shared sense above a
   matched-null baseline (McNemar's exact test). Inputs: GloVe + FastText `cc.en/cc.fr` +
   NLTK WordNet/OMW (`lang='fra'`).
3. **Cross-word semantic coherence (Fig. 2, §IV-D).** Pool the sense vectors in
   `data/polysemous_300words-glove-wiki-*.json`, cluster with no source-word labels, and show
   they group by semantic domain; a word's senses land in the correct different domains.
4. **BNC verb replication (§IV-E).** On POS-tagged, lemmatized BNC embeddings, SSR produces
   local sense-resolution refinement; quantified from `data/bnc_verbs_120_{baseline,ssr}.json`
   as the within-word vs. between-word distance ratio.

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
