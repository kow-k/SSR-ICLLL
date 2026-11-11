#!/usr/bin/env python3
"""
centroid_angles.py -- Section III-C / IV-A numbers for Spectral Sense Recovery.

For every word with expected_senses == 2 in the word list, discover two senses
UNSUPERVISED (spectral clustering of the word's neighbors at k=2) and report the
angle between the two sense vectors, for
  centroid : the cluster centroids (Spectral Sense Recovery, no repair)
  repair   : the same centroids after the original competitive repair
           (what discover_senses returned before)
No anchors or sense labels are used (unlike sense_localization_glove.py, which
uses hand-listed anchors per sense).

Also prints a worked example (default: bank): angle, top neighbors of each
centroid, and the nearest WordNet noun synset by gloss-word overlap is NOT
computed here -- use run_synset_mapping.py for synset assignment.

Usage (repository root):
  python centroid_angles.py glove-wiki-gigaword-100.gz data/polysemous_300words.csv
  python centroid_angles.py glove-wiki-gigaword-100.gz data/polysemous_300words.csv --example bank
"""
import argparse, csv, sys, os
import numpy as np

sys.path.insert(0, os.getcwd())
from sense_explorer import SenseExplorer, core


def angle(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("embedding")
    ap.add_argument("words_csv")
    ap.add_argument("--example", default="bank")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topn", type=int, default=8)
    a = ap.parse_args()

    np.random.seed(a.seed)
    se = SenseExplorer.from_glove(a.embedding, verbose=False)
    real_repair = core.SenseExplorer._simulated_repair
    ident = lambda self, w, c, n, sense_loyal=True: {
        s: v / (np.linalg.norm(v) + 1e-10) for s, v in c.items()}

    words = []
    with open(a.words_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            w = (r.get("word") or "").strip()
            if w and not w.startswith("#") and r.get("expected_senses", "").strip() == "2":
                words.append(w)
    if a.example not in words:
        words.insert(0, a.example)

    rows = []
    for idx, w in enumerate(words):
        if w not in se.vocab:
            continue
        out = {}
        for mode in ("centroid", "repair"):
            np.random.seed(a.seed + idx)   # identical clustering draw in both modes
            core.SenseExplorer._simulated_repair = ident if mode == "centroid" else real_repair
            s = se.discover_senses(w, n_senses=2, clustering_method="spectral", force=True)
            out[mode] = list(s.values())
        if len(out["centroid"]) == 2:
            rows.append((w, angle(*out["centroid"]),
                         angle(*out["repair"]) if len(out["repair"]) == 2 else float("nan"),
                         out["centroid"]))

    c = np.array([r[1] for r in rows]); rp = np.array([r[2] for r in rows])
    print(f"two-sense words with 2 recovered senses: n={len(rows)}")
    print(f"angle between sense vectors (degrees)")
    print(f"  centroid: mean={c.mean():.1f} median={np.median(c):.1f} "
          f"IQR=[{np.percentile(c,25):.1f},{np.percentile(c,75):.1f}]")
    ok = ~np.isnan(rp)
    print(f"  repair  : mean={rp[ok].mean():.1f} median={np.median(rp[ok]):.1f} "
          f"(n={ok.sum()}; competitive repair kept both senses)")

    ex = [r for r in rows if r[0] == a.example]
    if ex:
        w, ac, ar, vecs = ex[0]
        print(f"\nworked example '{w}': centroid angle={ac:.1f}  repair angle={ar:.1f}")
        E = se._embeddings_norm
        vocab = list(E.keys()) if isinstance(E, dict) else None
        M = np.array([E[v] for v in vocab])
        for i, v in enumerate(vecs):
            sims = M @ (v / np.linalg.norm(v))
            top = [vocab[j] for j in np.argsort(-sims) if vocab[j] != w][:a.topn]
            print(f"  sense {i}: {', '.join(top)}")


if __name__ == "__main__":
    main()
