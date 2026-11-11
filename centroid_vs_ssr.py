#!/usr/bin/env python3
"""
centroid_vs_ssr.py -- the reviewer's comparison: full SSR vs. its centroids alone.

Runs the cross-lingual recovery task of Table II (stage2_eval.py) and, on every
pair, scores three representations of the SAME clustering through the SAME
scorer:

  null      random partition of the same neighbors into k groups (as in the paper)
  centroid  the spectral clusters represented by their member centroids
            (= exactly what Table II scores)
  SSR       the package's own SenseExplorer._simulated_repair run from those
            centroids (sense-loyal, as in the paper's validations; competitive
            mode reported too)

Neighborhoods, the supplied sense count k = |S|, the clustering and the
Hungarian synset scoring are held constant; only the sense representation varies.
Reports, per neighborhood size:
  * recovery rate of each representation, and lift over the null
  * SSR vs centroid: McNemar's exact test on discordant pairs + bootstrap 95% CI
  * cos(SSR sense vector, its centroid) over all recovered sense vectors
    (the real-GloVe replacement for the paper's synthetic-probe figure 0.997)

Randomness: the global NumPy generator (used by spectral k-means) is seeded once
with --seed, exactly as a plain stage2 run would be; each repair runs on its own
temporarily swapped generator state, so adding SSR does not change the clusters.

Run from the repository root (needs sense_explorer/, stage1_anchor.py,
stage2_score_emb.py, io_utils.py, NLTK wordnet + omw):

  python centroid_vs_ssr.py --embedding glove.6B.100d.txt --topk 40,100
  python centroid_vs_ssr.py --embedding glove.6B.200d.txt --topk 40,100 --csv cvs_200d.csv
"""
import argparse
import csv
import gzip
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from io_utils import Emb, _unit_rows
from stage1_anchor import members, best_translation, pick_M, POS
from stage2_score_emb import assign_and_score_emb, syn_centroid, cl_centroid

MIN_NEIGHBORS = 8


# ---------------------------------------------------------------- loading
def load_raw(path, max_words):
    """Same reader as io_utils.load_glove (rank order, gensim header skipped),
    but keeps the raw vectors so SenseExplorer sees what it normally sees."""
    with open(path, "rb") as raw:
        magic = raw.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    words, rows = [], []
    with opener(path, "rt", encoding="utf8", errors="replace") as f:
        first = f.readline().rstrip().split(" ")
        if not (len(first) == 2 and all(t.isdigit() for t in first)):
            f.seek(0)
        for line in f:
            p = line.rstrip().split(" ")
            if len(p) < 3:
                continue
            words.append(p[0])
            rows.append(np.asarray(p[1:], dtype=np.float32))
            if len(words) >= max_words:
                break
    return words, np.vstack(rows)


# ---------------------------------------------------------------- scoring
def score_vectors(vecs, S, M, emb, cache):
    """assign_and_score_emb with the cluster centroids replaced by given vectors
    (None = a sense the method did not return). Identical logic otherwise."""
    Mj = [j for j, s in enumerate(S) if s.name() == M.name()][0]
    syn_c = [syn_centroid(cache[s.name()], emb) for s in S]
    if syn_c[Mj] is None:
        return False
    C, K = len(vecs), len(S)
    O = np.full((C, K), -1.0)
    for i, v in enumerate(vecs):
        if v is None:
            continue
        for j in range(K):
            if syn_c[j] is not None:
                O[i, j] = float(v @ syn_c[j])
    ri, ci = linear_sum_assignment(-O)
    amap = {int(r): int(c) for r, c in zip(ri, ci)}
    cl_for_M = [i for i, j in amap.items() if j == Mj]
    if not cl_for_M:
        return False
    i = cl_for_M[0]
    if vecs[i] is None:
        return False
    return int(np.argmax(O[i])) == Mj


def repair(explorer, word, cents, loyal, seed):
    """Package repair on an isolated RNG state; returns one vector (or None) per centroid."""
    saved = np.random.get_state()
    np.random.seed(seed)
    try:
        out = explorer._simulated_repair(
            word, {f"s{i}": c for i, c in enumerate(cents)},
            explorer.noise_level, sense_loyal=loyal)
    finally:
        np.random.set_state(saved)
    return [out.get(f"s{i}") for i in range(len(cents))]


# ---------------------------------------------------------------- main loop
def collect(emb, explorer, wn, top_k, max_pairs, cache, seed):
    """Mirror of stage2_eval.collect_records, plus centroid and SSR scoring."""
    from sense_explorer.spectral import spectral_clustering
    import random as _r
    pyrng = _r.Random(seed)
    nprng = np.random.default_rng(seed)
    names = sorted({l for l in wn.all_lemma_names(pos=POS) if l.isalpha() and len(l) >= 3})
    pyrng.shuffle(names)
    recs, cosines_loyal, cosines_comp, tilt = [], [], [], []
    mismatch = n = 0
    for x in names:
        if x not in emb.index:
            continue
        S = wn.synsets(x, pos=POS)
        if len(S) < 2:
            continue
        tr = best_translation(x, wn)
        if tr is None:
            continue
        y, shared, S = tr
        M = pick_M(shared)
        Ds = len(shared) / len(S)
        k = len(S)
        for s in S:
            cache.setdefault(s.name(), members(s))
        nbrs = emb.neighbors(x, top_k)
        if len(nbrs) < MIN_NEIGHBORS:
            continue
        V = np.array([emb.M[emb.index[w]] for w in nbrs])
        lab, _ = spectral_clustering(V, k=k)
        cl_s = [set() for _ in range(k)]
        for w, l in zip(nbrs, lab):
            cl_s[int(l)].add(w.lower())
        cl_s = [c for c in cl_s if c]
        rl = nprng.integers(0, k, len(nbrs))
        cl_r = [set() for _ in range(k)]
        for w, l in zip(nbrs, rl):
            cl_r[int(l)].add(w.lower())
        cl_r = [c for c in cl_r if c]
        if len(cl_s) < 2 or len(cl_r) < 2:
            continue

        # Table II scoring, unchanged
        r_cent, _ = assign_and_score_emb(cl_s, S, M, emb, cache)
        r_null, _ = assign_and_score_emb(cl_r, S, M, emb, cache)
        # same clusters as explicit vectors, through the vector scorer
        cents = [cl_centroid(c, emb) for c in cl_s]
        r_cent_v = score_vectors(cents, S, M, emb, cache)
        mismatch += int(bool(r_cent) != bool(r_cent_v))
        # full SSR from those centroids (package repair)
        usable = [c is not None for c in cents]
        cvec = [c for c in cents if c is not None]
        ssr_l = repair(explorer, x, cvec, True, seed + n)
        ssr_c = repair(explorer, x, cvec, False, seed + n)
        it_l, it_c = iter(ssr_l), iter(ssr_c)
        ssr_l = [next(it_l) if u else None for u in usable]
        ssr_c = [next(it_c) if u else None for u in usable]
        r_ssr_l = score_vectors(ssr_l, S, M, emb, cache)
        r_ssr_c = score_vectors(ssr_c, S, M, emb, cache)

        wvec = emb.M[emb.index[x]]
        for c, sl, sc in zip(cents, ssr_l, ssr_c):
            if c is None:
                continue
            if sl is not None:
                cosines_loyal.append(float(sl @ c))
                tilt.append(float(sl @ wvec) - float(c @ wvec))
            if sc is not None:
                cosines_comp.append(float(sc @ c))
        dropped = sum(1 for c, s in zip(cents, ssr_c) if c is not None and s is None)
        recs.append(dict(word=x, k=k, n_clusters=len(cl_s), D_s=Ds,
                         null=bool(r_null), centroid=bool(r_cent),
                         ssr_loyal=bool(r_ssr_l), ssr_competitive=bool(r_ssr_c),
                         competitive_dropped=dropped))
        n += 1
        if max_pairs and n >= max_pairs:
            break
    return recs, np.array(cosines_loyal), np.array(cosines_comp), np.array(tilt), mismatch


# ---------------------------------------------------------------- statistics
def mcnemar(a, b):
    from scipy.stats import binomtest
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    return n10, n01, p


def boot_diff(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(a)
    idx = rng.integers(0, m, (n, m))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def report(recs, cl, cc, tilt, mismatch, label):
    g = lambda key: np.array([r[key] for r in recs])
    null, cent, ssr, comp = g("null"), g("centroid"), g("ssr_loyal"), g("ssr_competitive")
    print(f"\n[{label}]  n = {len(recs)} pairs")
    print(f"  recovery   null={null.mean():.3f}  centroid={cent.mean():.3f}  "
          f"SSR(loyal)={ssr.mean():.3f}  SSR(competitive)={comp.mean():.3f}")
    print(f"  lift over null   centroid={cent.mean()-null.mean():+.3f}  "
          f"SSR(loyal)={ssr.mean()-null.mean():+.3f}  SSR(competitive)={comp.mean()-null.mean():+.3f}")
    for name, arr in [("SSR(loyal)", ssr), ("SSR(competitive)", comp)]:
        n10, n01, p = mcnemar(arr, cent)
        lo, hi = boot_diff(arr, cent)
        print(f"  {name} - centroid = {arr.mean()-cent.mean():+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  "
              f"McNemar: SSR-only={n10} centroid-only={n01} p={p:.3g}")
    if len(cl):
        print(f"  cos(SSR loyal, centroid): mean={cl.mean():.4f}  median={np.median(cl):.4f}  "
              f"1st pct={np.percentile(cl,1):.4f}  min={cl.min():.4f}  (n={len(cl)} sense vectors)")
    if len(cc):
        print(f"  cos(SSR competitive, centroid): mean={cc.mean():.4f}  min={cc.min():.4f}  (n={len(cc)})")
    if len(tilt):
        print(f"  tilt toward the word's own vector, cos(s,w)-cos(c,w): mean={tilt.mean():+.4f}")
    dropped = sum(r["competitive_dropped"] for r in recs)
    print(f"  competitive mode dropped {dropped} sense(s) in "
          f"{sum(1 for r in recs if r['competitive_dropped'])} pairs")
    print(f"  sanity: vector scorer disagrees with Table II scorer on {mismatch} pairs (expect 0)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--topk", default="40,100", help="neighborhood sizes (Table II: 40,100)")
    ap.add_argument("--max-pairs", type=int, default=1500)
    ap.add_argument("--max-words", type=int, default=60000, help="as stage2_eval (60000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None, help="per-pair outcomes (all top_k) to this file")
    args = ap.parse_args()

    import nltk
    for r in ("wordnet", "omw-1.4"):
        nltk.download(r, quiet=True)
    from nltk.corpus import wordnet as wn
    wn.synsets("dog")

    from sense_explorer import SenseExplorer

    words, R = load_raw(args.embedding, args.max_words)
    emb = Emb(words, _unit_rows(R))
    explorer = SenseExplorer({w: R[i] for i, w in enumerate(words)},
                             verbose=False, use_hybrid_anchors=False)
    print(f"loaded {len(words)} words x {R.shape[1]}d from {args.embedding}")
    print(f"SSR repair settings: n_copies={explorer.n_copies} noise={explorer.noise_level} "
          f"seed_strength={explorer.seed_strength} n_iter={explorer.n_iterations} "
          f"anchor_pull={explorer.anchor_pull}")

    np.random.seed(args.seed)          # spectral k-means stream, as in a seeded stage2 run
    cache, rows = {}, []
    for tk in [int(t) for t in args.topk.split(",")]:
        recs, cl, cc, tilt, mm = collect(emb, explorer, wn, tk, args.max_pairs, cache, args.seed)
        report(recs, cl, cc, tilt, mm, f"top_k={tk}")
        for r in recs:
            r["top_k"] = tk
        rows.extend(recs)

    if args.csv:
        cols = ["top_k", "word", "k", "n_clusters", "D_s", "null", "centroid",
                "ssr_loyal", "ssr_competitive", "competitive_dropped"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r[c] for c in cols})
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
