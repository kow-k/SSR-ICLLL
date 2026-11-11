#!/usr/bin/env python3
"""
noise_only_ssr.py -- SSR without neighbors: does clustering noisy copies of a
word vector recover its senses?

Method under test (never implemented before):
    e                      the word's vector
    C  = {c_1..c_N}        N copies of e
    C' = {c'_1..c'_N}      each copy perturbed by independent random noise
    spectral clustering of C' into k groups (same spectral_clustering as SSR)
    sense vector s_i = normalized mean of the copies in group i
No other word of the vocabulary is consulted.

Noise models
  package    the SenseExplorer noise: a random 50-80% of dimensions per copy,
             N(0, noise * |e_d|) on each perturbed dimension d
  isotropic  N(0, noise * ||e|| / sqrt(d)) on every dimension

Evaluation: the cross-lingual task and scorer of Table II (as centroid_vs_ssr.py).
On every pair, with the same supplied k = |S|, it scores
  null       random partition of the word's neighbors (Table II's null)
  neighbors  spectral clusters of the word's neighbors (Table II's SSR column)
  e_alone    k copies of e itself, no noise: recovered iff e's own nearest synset
             centroid is the shared synset M (what a method that adds nothing
             beyond e should score)
  noise:...  noise-only SSR, one column per configuration
and reports recovery, McNemar tests (noise-only vs e_alone and vs null), and
geometry: cos(sense vector, e) and the mean pairwise cosine between the k sense
vectors of a word (1.0 = all identical, i.e. nothing separated).

  --demo bank,mouse,cell   prints, for each word, the nearest vocabulary words
                           of every noise-only sense vector next to those of
                           the neighbor-based clusters (k = 2 unless given as word:k)

Run from the repository root (needs sense_explorer/, stage1_anchor.py,
stage2_score_emb.py, io_utils.py, NLTK wordnet + omw):
  python noise_only_ssr.py --embedding glove.6B.100d.txt --demo bank,mouse,cell,plant,crane,spring:3
  python noise_only_ssr.py --embedding glove.6B.100d.txt --topk 40 --csv nos_100d.csv
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
from sense_explorer.spectral import spectral_clustering

MIN_NEIGHBORS = 8
CONFIGS = [  # (model, noise, N)
    ("package", 0.5, 30), ("package", 0.5, 300),
    ("package", 1.0, 300), ("package", 2.0, 300),
    ("isotropic", 0.5, 300), ("isotropic", 1.0, 300), ("isotropic", 2.0, 300),
]


# ---------------------------------------------------------------- method
def noisy_copies(e, N, noise, model, rng):
    d = len(e)
    C = np.tile(e, (N, 1)).astype(np.float64)
    if model == "package":
        frac = rng.uniform(0.5, 0.8, (N, 1))
        mask = rng.uniform(0, 1, (N, d)) < frac
        C += rng.standard_normal((N, d)) * np.abs(e) * noise * mask
    elif model == "isotropic":
        C += rng.standard_normal((N, d)) * (noise * np.linalg.norm(e) / np.sqrt(d))
    else:
        raise ValueError(model)
    return C


def noise_only_senses(e, k, N, noise, model, rng):
    """Noise-only SSR: k sense vectors (unit), or None for an empty group."""
    Cp = noisy_copies(e, N, noise, model, rng)
    state = np.random.get_state()            # spectral k-means uses the global RNG
    np.random.seed(int(rng.integers(2**31)))
    try:
        lab, _ = spectral_clustering(Cp, k=k)
    finally:
        np.random.set_state(state)
    out = []
    for i in range(k):
        g = Cp[lab == i]
        out.append(None if len(g) == 0 else g.mean(0) / (np.linalg.norm(g.mean(0)) + 1e-12))
    return out


# ---------------------------------------------------------------- loading / scoring
def load_raw(path, max_words):
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


def score_vectors(vecs, S, M, emb, cache):
    """Table II scorer with the cluster centroids replaced by given vectors."""
    Mj = [j for j, s in enumerate(S) if s.name() == M.name()][0]
    syn_c = [syn_centroid(cache[s.name()], emb) for s in S]
    if syn_c[Mj] is None:
        return False
    O = np.full((len(vecs), len(S)), -1.0)
    for i, v in enumerate(vecs):
        if v is None:
            continue
        for j in range(len(S)):
            if syn_c[j] is not None:
                O[i, j] = float(v @ syn_c[j])
    ri, ci = linear_sum_assignment(-O)
    cl_for_M = [int(r) for r, c in zip(ri, ci) if int(c) == Mj]
    if not cl_for_M or vecs[cl_for_M[0]] is None:
        return False
    return int(np.argmax(O[cl_for_M[0]])) == Mj


def cfg_name(c):
    return f"noise:{c[0]}/{c[1]}/N{c[2]}"


def geometry(vecs, e_unit):
    v = [x for x in vecs if x is not None]
    if len(v) < 2:
        return np.nan, np.nan
    V = np.array(v)
    G = V @ V.T
    iu = np.triu_indices(len(v), 1)
    return float(np.mean(V @ e_unit)), float(np.mean(G[iu]))


# ---------------------------------------------------------------- task
def collect(emb, R, words_index, wn, top_k, max_pairs, cache, seed):
    import random as _r
    pyrng = _r.Random(seed)
    nprng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 12345)
    names = sorted({l for l in wn.all_lemma_names(pos=POS) if l.isalpha() and len(l) >= 3})
    pyrng.shuffle(names)
    recs, geo = [], {cfg_name(c): [] for c in CONFIGS}
    geo["neighbors"] = []
    n = 0
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

        e_raw = R[words_index[x]].astype(np.float64)
        e_unit = e_raw / np.linalg.norm(e_raw)
        rec = dict(word=x, k=k)
        rec["null"], _ = assign_and_score_emb(cl_r, S, M, emb, cache)
        rec["neighbors"], _ = assign_and_score_emb(cl_s, S, M, emb, cache)
        rec["e_alone"] = score_vectors([e_unit] * k, S, M, emb, cache)
        geo["neighbors"].append(geometry([cl_centroid(c, emb) for c in cl_s], e_unit))
        for c in CONFIGS:
            vecs = noise_only_senses(e_raw, k, c[2], c[1], c[0], noise_rng)
            rec[cfg_name(c)] = score_vectors(vecs, S, M, emb, cache)
            geo[cfg_name(c)].append(geometry(vecs, e_unit))
        recs.append({kk: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for kk, v in rec.items()})
        n += 1
        if max_pairs and n >= max_pairs:
            break
    return recs, geo


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    return n10, n01, p


def report(recs, geo, label):
    g = lambda key: np.array([r[key] for r in recs], dtype=bool)
    null, nb, ea = g("null"), g("neighbors"), g("e_alone")
    print(f"\n[{label}]  n = {len(recs)} pairs   (recovery of the shared synset; higher is better)")
    print(f"  {'method':32s} {'recovery':>8s} {'vs null':>8s} {'vs e_alone':>11s}  "
          f"{'McNemar vs e_alone (only/only, p)':>34s}  {'cos(s,e)':>8s} {'cos(s_i,s_j)':>12s}")
    rows = [("null (random neighbor partition)", null, None),
            ("neighbors (Table II SSR)", nb, "neighbors"),
            ("e_alone (k copies of e)", ea, None)] + \
           [(cfg_name(c), g(cfg_name(c)), cfg_name(c)) for c in CONFIGS]
    for name, arr, gk in rows:
        n10, n01, p = mcnemar(arr, ea)
        if gk:
            G = np.array(geo[gk], dtype=float)
            ce, cc = np.nanmean(G[:, 0]), np.nanmean(G[:, 1])
            gtxt = f"{ce:8.3f} {cc:12.3f}"
        else:
            gtxt = f"{'':8s} {'':12s}"
        print(f"  {name:32s} {arr.mean():8.3f} {arr.mean()-null.mean():+8.3f} {arr.mean()-ea.mean():+11.3f}  "
              f"{f'{n10}/{n01}, p={p:.2g}':>34s}  {gtxt}")


# ---------------------------------------------------------------- demo
def demo(emb, R, words_index, spec, top_k, seed, nn=8):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    M = emb.M
    for item in spec.split(","):
        w, _, kk = item.partition(":")
        k = int(kk) if kk else 2
        if w not in words_index:
            print(f"\n{w}: not in vocabulary"); continue
        e_raw = R[words_index[w]].astype(np.float64)
        e_unit = e_raw / np.linalg.norm(e_raw)

        def show(vecs, title):
            ce, cc = geometry(vecs, e_unit)
            print(f"  {title}   cos(s,e)={ce:.3f}  cos between senses={cc:.3f}")
            for i, v in enumerate(vecs):
                if v is None:
                    print(f"    s{i}: (empty)"); continue
                sims = M @ (v / np.linalg.norm(v))
                top = [emb.words[j] for j in np.argsort(-sims)[: nn + 1] if emb.words[j] != w][:nn]
                print(f"    s{i}: {', '.join(top)}")

        print(f"\n=== {w}  (k={k}) ===")
        nbrs = emb.neighbors(w, top_k)
        V = np.array([M[emb.index[x]] for x in nbrs])
        lab, _ = spectral_clustering(V, k=k)
        show([cl_centroid({x for x, l in zip(nbrs, lab) if l == i}, emb) for i in range(k)],
             f"neighbors (top_k={top_k})")
        for c in CONFIGS:
            show(noise_only_senses(e_raw, k, c[2], c[1], c[0], rng), cfg_name(c))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--topk", default="40", help="neighborhood size(s) for the null and neighbor columns")
    ap.add_argument("--max-pairs", type=int, default=1500)
    ap.add_argument("--max-words", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", default=None, help="comma-separated words, e.g. bank,mouse,spring:3")
    ap.add_argument("--no-task", action="store_true", help="only run --demo")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    words, R = load_raw(args.embedding, args.max_words)
    emb = Emb(words, _unit_rows(R))
    if not hasattr(emb, "words"):
        emb.words = words
    words_index = {w: i for i, w in enumerate(words)}
    print(f"loaded {len(words)} words x {R.shape[1]}d from {args.embedding}")
    print("noise-only SSR configurations (model/noise/N): " + ", ".join(cfg_name(c)[6:] for c in CONFIGS))

    if args.demo:
        demo(emb, R, words_index, args.demo, int(args.topk.split(",")[0]), args.seed)
    if args.no_task:
        return 0

    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.synsets("chien", lang="fra")
    except LookupError:
        for r in ("wordnet", "omw-1.4"):
            nltk.download(r, quiet=True)
        wn.synsets("dog")

    np.random.seed(args.seed)
    cache, rows = {}, []
    for tk in [int(t) for t in args.topk.split(",")]:
        recs, geo = collect(emb, R, words_index, wn, tk, args.max_pairs, cache, args.seed)
        report(recs, geo, f"top_k={tk}")
        for r in recs:
            r["top_k"] = tk
        rows.extend(recs)
    if args.csv and rows:
        cols = list(rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
