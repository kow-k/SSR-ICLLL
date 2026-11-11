#!/usr/bin/env python3
"""
table2_swap.py -- Table II with a swap null.

Table II scores the shared-sense recovery of SSR's neighbor clusters against a
null that partitions the same neighbors at random. Random partitions have
near-identical centroids, so they can be scored lower than ANY compact clustering
regardless of meaning. This script adds a null that keeps the clustering and
breaks only its link to the word:

  swap   the SAME clusters of target x are scored, with the Table II rule, against
         the synset inventory (S_y, shared synset M_y) of another target y with the
         same sense count |S_y| = |S_x| and no synset in common.

Scoring note: Table II's synset member sets include each synset's own lemmas, i.e.
the target word itself, which the target's neighbor clusters are built around. Part
(B) therefore re-scores everything with the inventory's own target word removed
from its synset centroids (x for x's synsets, y for y's), which makes SSR, the
random-partition null and the swap null directly comparable. Part (A) is the
published scoring.

For every pair it reports Table II's own columns (SSR, random-partition null), the
swap null, and the lifts of SSR over both (McNemar on paired outcomes + bootstrap
95% CI). The pair set, neighbors, k = |S| and clustering are exactly Table II's
(same seed and global RNG stream), so the SSR and random-null columns reproduce
Table II.

Run from the repository root (needs stage1_anchor.py, stage2_score_emb.py,
io_utils.py, sense_explorer/, NLTK wordnet + omw):
  python table2_swap.py --embedding glove.6B.100d.txt --topk 40,100 --csv t2swap_100d.csv
  python table2_swap.py --embedding glove.6B.100d.txt --pool dedup --show-pool bank,spring,bat --topk 50
  python table2_swap.py --embedding glove.6B.100d.txt --pool dedup --dedup-q 0.9 --topk 30,50,70,100
"""
import argparse
import csv
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from io_utils import Emb, _unit_rows
from stage1_anchor import members, best_translation, pick_M, POS
from stage2_score_emb import assign_and_score_emb, syn_centroid, cl_centroid
from sense_explorer.spectral import spectral_clustering
from adaptive_k import load_raw

MIN_NEIGHBORS = 8


def score_on(cents, S, M, emb, cache, exclude=None):
    """Table II rule for given cluster centroids against inventory (S, M).
    exclude: a word left out of every synset's member set (the inventory's own target)."""
    Mj = [j for j, s in enumerate(S) if s.name() == M.name()][0]
    syn_c = [syn_centroid(cache[s.name()] - ({exclude} if exclude else set()), emb) for s in S]
    if syn_c[Mj] is None:
        return False
    O = np.full((len(cents), len(S)), -1.0)
    for i, c in enumerate(cents):
        if c is None:
            continue
        for j in range(len(S)):
            if syn_c[j] is not None:
                O[i, j] = float(c @ syn_c[j])
    ri, ci = linear_sum_assignment(-O)
    hit = [int(r) for r, c in zip(ri, ci) if int(c) == Mj]
    return bool(hit) and cents[hit[0]] is not None and int(np.argmax(O[hit[0]])) == Mj


_TR = {}
POOL = dict(mode="none", q=0.9, lam=0.7, factor=5)


def build_pool(emb, x, K):
    """The K-word neighbor pool of x, optionally de-duplicated (no labels used).
    none   top-K by cosine (Table II)
    dedup  walk down the ranking; skip a word whose cosine to an already kept word exceeds
           the q-quantile of pairwise cosines within x's plain top-K pool
    mmr    greedy maximal marginal relevance: argmax lam*cos(c,x) - (1-lam)*max cos(c,kept)"""
    if POOL["mode"] == "none":
        return emb.neighbors(x, K)
    cand = emb.neighbors(x, K * POOL["factor"])
    C = np.array([emb.M[emb.index[w]] for w in cand])
    if POOL["mode"] == "dedup":
        P = C[:K] @ C[:K].T
        thr = np.quantile(P[np.triu_indices(K, 1)], POOL["q"])
        kept = [0]
        for i in range(1, len(cand)):
            if len(kept) >= K:
                break
            if np.max(C[kept] @ C[i]) <= thr:
                kept.append(i)
        return [cand[i] for i in kept]
    if POOL["mode"] == "mmr":
        rel = C @ emb.M[emb.index[x]]
        kept = [0]; maxsim = C @ C[0]
        avail = np.ones(len(cand), bool); avail[0] = False
        while len(kept) < K and avail.any():
            score = POOL["lam"] * rel - (1 - POOL["lam"]) * maxsim
            score[~avail] = -np.inf
            i = int(np.argmax(score)); kept.append(i); avail[i] = False
            maxsim = np.maximum(maxsim, C @ C[i])
        return [cand[i] for i in kept]
    raise ValueError(POOL["mode"])


def collect(emb, wn, top_k, max_pairs, cache, seed):
    import random as _r
    pyrng = _r.Random(seed)
    nprng = np.random.default_rng(seed)
    names = sorted({l for l in wn.all_lemma_names(pos=POS) if l.isalpha() and len(l) >= 3})
    pyrng.shuffle(names)
    recs, n = [], 0
    for x in names:
        if x not in emb.index:
            continue
        S = wn.synsets(x, pos=POS)
        if len(S) < 2:
            continue
        if x not in _TR:
            _TR[x] = best_translation(x, wn)
        tr = _TR[x]
        if tr is None:
            continue
        y, shared, S = tr
        M = pick_M(shared)
        k = len(S)
        for s in S:
            cache.setdefault(s.name(), members(s))
        nbrs = build_pool(emb, x, top_k)
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
        cents = [cl_centroid(c, emb) for c in cl_s]
        rcents = [cl_centroid(c, emb) for c in cl_r]
        recs.append(dict(word=x, k=k, S=S, M=M, cents=cents,
                         ssr_x=score_on(cents, S, M, emb, cache, exclude=x),
                         null_x=score_on(rcents, S, M, emb, cache, exclude=x),
                         ssr=bool(assign_and_score_emb(cl_s, S, M, emb, cache)[0]),
                         null=bool(assign_and_score_emb(cl_r, S, M, emb, cache)[0])))
        n += 1
        if max_pairs and n >= max_pairs:
            break
    return recs


def add_swap(recs, emb, cache, seed):
    rng = np.random.default_rng(seed + 4242)
    byk = {}
    for i, r in enumerate(recs):
        byk.setdefault(r["k"], []).append(i)
    for i, r in enumerate(recs):
        names = {s.name() for s in r["S"]}
        cand = [j for j in byk[r["k"]] if j != i and not (names & {s.name() for s in recs[j]["S"]})]
        if not cand:
            r["swap"] = None; r["swap_word"] = ""
            continue
        j = cand[rng.integers(len(cand))]
        r["swap"] = score_on(r["cents"], recs[j]["S"], recs[j]["M"], emb, cache, exclude=recs[j]["word"])
        r["swap_word"] = recs[j]["word"]


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    return n10, n01, p


def boot(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


SUMMARY = []


def report(recs, label):
    R = [r for r in recs if r["swap"] is not None]
    g = lambda key: np.array([r[key] for r in R])
    a, b = g("ssr"), g("null")
    ax, bx, c = g("ssr_x"), g("null_x"), g("swap")
    print(f"\n[{label}]  n = {len(R)} pairs (of {len(recs)}; the rest had no swap partner)")
    def line(name, u, z):
        n10, n01, p = mcnemar(u, z); lo, hi = boot(u, z)
        print(f"    SSR - {name:26s} {u.mean()-z.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  McNemar {n10}/{n01}, p={p:.2g}")
    print(f"  (A) published scoring (target word inside its own synset centroids)")
    print(f"      recovery  SSR {a.mean():.3f}   random-partition null {b.mean():.3f}")
    line("random-partition null", a, b)
    print(f"  (B) target excluded from synset centroids (symmetric with the swap null)")
    print(f"      recovery  SSR {ax.mean():.3f}   random-partition null {bx.mean():.3f}   swap null {c.mean():.3f}")
    line("random-partition null", ax, bx)
    line("swap null", ax, c)
    lo, hi = boot(ax, c); n10, n01, p = mcnemar(ax, c)
    SUMMARY.append((label, a.mean() - b.mean(), ax.mean() - c.mean(), lo, hi, p, len(R)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--topk", default="40,100")
    ap.add_argument("--max-pairs", type=int, default=1500)
    ap.add_argument("--max-words", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--pool", choices=["none", "dedup", "mmr"], default="none",
                    help="neighbor-pool cleaning (unsupervised): none = Table II")
    ap.add_argument("--dedup-q", type=float, default=0.9,
                    help="dedup: skip words closer to a kept word than this quantile of the plain pool's pairwise cosines")
    ap.add_argument("--mmr-lambda", type=float, default=0.7, help="mmr: weight on closeness to the target")
    ap.add_argument("--show-pool", default=None, help="comma-separated words: print their plain and cleaned pools and exit")
    args = ap.parse_args()
    POOL.update(mode=args.pool, q=args.dedup_q, lam=args.mmr_lambda)

    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.synsets("chien", lang="fra")
    except LookupError:
        for r in ("wordnet", "omw-1.4"):
            nltk.download(r, quiet=True)

    words, Rm = load_raw(args.embedding, args.max_words)
    emb = Emb(words, _unit_rows(Rm))
    print(f"loaded {len(words)} words x {Rm.shape[1]}d from {args.embedding}")
    print(f"pool: {args.pool}" + (f" (q={args.dedup_q})" if args.pool == "dedup" else
                                  f" (lambda={args.mmr_lambda})" if args.pool == "mmr" else ""))
    if args.show_pool:
        K = int(args.topk.split(",")[0])
        for w in args.show_pool.split(","):
            mode = POOL["mode"]; POOL["mode"] = "none"; plain = build_pool(emb, w, K); POOL["mode"] = mode
            clean = build_pool(emb, w, K)
            print(f"\n{w}  (K={K})\n  plain  : {', '.join(plain)}\n  {mode:7s}: {', '.join(clean)}")
            print(f"  new in cleaned pool: {', '.join(x for x in clean if x not in plain)}")
        return 0
    np.random.seed(args.seed)
    cache, rows = {}, []
    for tk in [int(t) for t in args.topk.split(",")]:
        recs = collect(emb, wn, tk, args.max_pairs, cache, args.seed)
        add_swap(recs, emb, cache, args.seed)
        report(recs, f"K={tk}")
        sys.stdout.flush()
        for r in recs:
            rows.append(dict(K=tk, word=r["word"], k=r["k"], ssr=r["ssr"], null=r["null"],
                             ssr_x=r["ssr_x"], null_x=r["null_x"], swap=r["swap"], swap_word=r["swap_word"]))
    print(f"\n=== summary: {args.embedding} ===")
    print(f"  {'K':>6s} {'published lift':>15s} {'swap lift':>10s} {'95% CI':>18s} {'p':>8s}")
    for label, pub, sw, lo, hi, p, n in SUMMARY:
        print(f"  {label:>6s} {pub:+15.3f} {sw:+10.3f}   [{lo:+.3f}, {hi:+.3f}] {p:8.2g}")
    sw = np.array([x[2] for x in SUMMARY])
    print(f"  mean swap lift across K: {sw.mean():+.3f}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
