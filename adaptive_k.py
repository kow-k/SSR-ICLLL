#!/usr/bin/env python3
"""
adaptive_k.py -- choose the sense count k from the neighbor pool itself, and test it
on the cross-lingual recovery task of Table II.

For every pair of the Table II task (same pairs, same neighbors, same scorer),
the target's K nearest neighbors are partitioned into k groups, where k is
chosen by one of these rules -- none of which sees WordNet:

  eigengap_current  the package rule: largest Laplacian eigengap, k in 2..5
                    (what discover_senses_auto does today; can never say 1)
  eigengap_k1       the same rule but k in 1..MAX_K: a single sense is allowed
  null_calibrated   k = 1 + the number of low Laplacian eigenvalues (lambda_1,
                    lambda_2, ... in order) that fall below the 5th percentile of
                    the same eigenvalue in structureless control pools. A control
                    pool permutes each embedding dimension independently across
                    the K neighbors: same per-dimension spread, one blob, no groups.
  stability         for each k >= 2, cluster random 80% subsamples in pairs and
                    measure agreement (adjusted Rand index on the overlap); the
                    same on a control pool. k = the k whose stability exceeds the
                    control's by the most, if by more than 2 standard errors;
                    otherwise k = 1.

Reference rows (these do use or fix the count):
  wordnet_S   k = |S|, the WordNet synset count -- exactly Table II's SSR column
  fixed_2     k = 2 for every word

Each rule is scored against a random partition of the same neighbors into the
SAME k it chose (its own matched null), so choosing more clusters cannot buy
recovery. With k = 1, method and null coincide (lift 0 for that pair).

Reported per rule: recovery, matched-null recovery, lift and McNemar's exact test,
the distribution of chosen k, how often k = |S|, and mean |S| for comparison.

Run from the repository root (needs sense_explorer/, stage1_anchor.py,
stage2_score_emb.py, io_utils.py, NLTK wordnet + omw):
  python adaptive_k.py --embedding glove.6B.100d.txt --topk 40,100 --csv ak_100d.csv
  python adaptive_k.py --embedding glove.6B.100d.txt --demo bank,mouse,cell,plant,crane,spring
"""
import argparse
import csv
import gzip
import sys
from collections import Counter

import numpy as np

from io_utils import Emb, _unit_rows
from stage1_anchor import members, best_translation, pick_M, POS
from stage2_score_emb import assign_and_score_emb
from sense_explorer.spectral import (spectral_clustering, build_similarity_matrix,
                                     compute_laplacian_eigenvectors, find_k_by_eigengap,
                                     _kmeans_simple)

MIN_NEIGHBORS = 8
MAX_K = 6
RULES = ["eigengap_current", "eigengap_k1", "null_calibrated", "stability"]
REFS = ["wordnet_S", "fixed_2"]


# ---------------------------------------------------------------- spectral pieces
def unit(V):
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-10)


def spectrum(V, m):
    vals, vecs = compute_laplacian_eigenvectors(build_similarity_matrix(unit(V)), min(m, len(V)))
    return vals, vecs


def partition(V, k):
    """Same as spectral_clustering(V, k=k), plus k = 1."""
    if k <= 1:
        return np.zeros(len(V), dtype=int)
    lab, _ = spectral_clustering(V, k=k)
    return lab


def control_pool(V, rng):
    """Permute every dimension independently across the pool: no group structure."""
    P = V.copy()
    for d in range(P.shape[1]):
        P[:, d] = P[rng.permutation(len(P)), d]
    return P


def ari(a, b):
    """Adjusted Rand index of two labelings."""
    a = np.asarray(a); b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    C = np.zeros((len(ua), len(ub)))
    np.add.at(C, (ia, ib), 1)
    c2 = lambda x: x * (x - 1) / 2.0
    s = c2(C).sum(); sa = c2(C.sum(1)).sum(); sb = c2(C.sum(0)).sum(); n = c2(len(a))
    exp = sa * sb / n if n else 0.0
    mx = 0.5 * (sa + sb)
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


# ---------------------------------------------------------------- k rules
def k_eigengap_current(vals):
    return find_k_by_eigengap(vals, 2, 5)


def k_eigengap_k1(vals):
    gaps = np.diff(vals[: MAX_K + 1])
    return int(np.argmax(gaps[:MAX_K])) + 1


def k_null_calibrated(V, vals, rng, R):
    null = np.array([spectrum(control_pool(V, rng), MAX_K + 1)[0] for _ in range(R)])
    q = np.percentile(null, 5, axis=0)
    k = 1
    for i in range(1, min(MAX_K, len(vals))):
        if vals[i] < q[i]:
            k += 1
        else:
            break
    return k


def stability_scores(V, rng, B):
    n = len(V); m = max(int(round(0.8 * n)), MAX_K + 2)
    out = {}
    for k in range(2, MAX_K + 1):
        vals = []
        for _ in range(B):
            i1 = rng.choice(n, m, replace=False); i2 = rng.choice(n, m, replace=False)
            l1 = partition(V[i1], k); l2 = partition(V[i2], k)
            common, a1, a2 = np.intersect1d(i1, i2, return_indices=True)
            vals.append(ari(l1[a1], l2[a2]) if len(common) > k else 0.0)
        out[k] = np.array(vals)
    return out


def k_stability(V, rng, B):
    real = stability_scores(V, rng, B)
    ctrl = stability_scores(control_pool(V, rng), rng, B)
    best, best_diff = 1, 0.0
    for k in range(2, MAX_K + 1):
        d = real[k].mean() - ctrl[k].mean()
        se = np.sqrt(real[k].var(ddof=1) / B + ctrl[k].var(ddof=1) / B) + 1e-12
        if d > 2 * se and d > best_diff:
            best, best_diff = k, d
    return best


def choose_all(V, rng, R, B):
    vals, _ = spectrum(V, MAX_K + 2)
    return {"eigengap_current": k_eigengap_current(vals),
            "eigengap_k1": k_eigengap_k1(vals),
            "null_calibrated": k_null_calibrated(V, vals, rng, R),
            "stability": k_stability(V, rng, B)}


# ---------------------------------------------------------------- loading
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


def to_sets(nbrs, lab, k):
    cl = [set() for _ in range(max(k, 1))]
    for w, l in zip(nbrs, lab):
        cl[int(l)].add(w.lower())
    return [c for c in cl if c]


# ---------------------------------------------------------------- task
def collect(emb, wn, top_k, max_pairs, cache, seed, R, B):
    import random as _r
    pyrng = _r.Random(seed)
    nprng = np.random.default_rng(seed)          # Table II null stream (unchanged)
    rule_rng = np.random.default_rng(seed + 777)  # controls, subsamples, matched nulls
    names = sorted({l for l in wn.all_lemma_names(pos=POS) if l.isalpha() and len(l) >= 3})
    pyrng.shuffle(names)
    recs, n = [], 0
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
        kS = len(S)
        for s in S:
            cache.setdefault(s.name(), members(s))
        nbrs = emb.neighbors(x, top_k)
        if len(nbrs) < MIN_NEIGHBORS:
            continue
        V = np.array([emb.M[emb.index[w]] for w in nbrs])

        # ---- Table II exactly (global RNG stream), defines the pair set
        lab, _ = spectral_clustering(V, k=kS)
        cl_s = to_sets(nbrs, lab, kS)
        rl = nprng.integers(0, kS, len(nbrs))
        cl_r = to_sets(nbrs, rl, kS)
        if len(cl_s) < 2 or len(cl_r) < 2:
            continue
        rec = dict(word=x, S=kS)
        rec["wordnet_S_k"] = kS
        rec["wordnet_S"] = bool(assign_and_score_emb(cl_s, S, M, emb, cache)[0])
        rec["wordnet_S_null"] = bool(assign_and_score_emb(cl_r, S, M, emb, cache)[0])

        # ---- everything else on an isolated, reproducible RNG state
        saved = np.random.get_state()
        np.random.seed((seed * 1000003 + n) % (2**31))
        try:
            ks = choose_all(V, rule_rng, R, B)
            ks["fixed_2"] = 2
            for name, k in ks.items():
                lab = partition(V, k)
                cl = to_sets(nbrs, lab, k)
                rnd = rule_rng.integers(0, k, len(nbrs)) if k > 1 else np.zeros(len(nbrs), int)
                cln = to_sets(nbrs, rnd, k)
                rec[f"{name}_k"] = k
                rec[name] = bool(assign_and_score_emb(cl, S, M, emb, cache)[0])
                rec[f"{name}_null"] = bool(assign_and_score_emb(cln, S, M, emb, cache)[0])
        finally:
            np.random.set_state(saved)
        recs.append(rec)
        n += 1
        if n % 100 == 0:
            print(f"  ... {n} pairs", file=sys.stderr)
        if max_pairs and n >= max_pairs:
            break
    return recs


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    return n10, n01, p


def report(recs, label):
    S = np.array([r["S"] for r in recs])
    print(f"\n[{label}]  n = {len(recs)} pairs;  WordNet |S|: mean {S.mean():.2f}, median {np.median(S):.0f}")
    print(f"  {'rule':18s} {'recov':>6s} {'null':>6s} {'lift':>7s} {'McNemar (only/only, p)':>24s}"
          f" {'mean k':>6s} {'k=1':>5s} {'k=|S|':>6s}   k distribution 1..{MAX_K}+")
    for name in REFS[:1] + RULES + REFS[1:]:
        a = np.array([r[name] for r in recs]); b = np.array([r[f"{name}_null"] for r in recs])
        k = np.array([r[f"{name}_k"] for r in recs])
        n10, n01, p = mcnemar(a, b)
        dist = Counter(np.minimum(k, MAX_K))
        dtxt = " ".join(f"{dist.get(i, 0):4d}" for i in range(1, MAX_K + 1))
        print(f"  {name:18s} {a.mean():6.3f} {b.mean():6.3f} {a.mean()-b.mean():+7.3f} "
              f"{f'{n10}/{n01}, p={p:.2g}':>24s} {k.mean():6.2f} {np.mean(k==1):5.2f} {np.mean(k==S):6.2f}   {dtxt}")


# ---------------------------------------------------------------- demo
def demo(emb, spec, top_k, seed, R, B, nn=8):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    for w in spec.split(","):
        if w not in emb.index:
            print(f"\n{w}: not in vocabulary"); continue
        nbrs = emb.neighbors(w, top_k)
        V = np.array([emb.M[emb.index[x]] for x in nbrs])
        ks = choose_all(V, rng, R, B)
        vals, _ = spectrum(V, MAX_K + 2)
        print(f"\n=== {w}  (K={top_k}) ===  chosen k: " + ", ".join(f"{r}={k}" for r, k in ks.items()))
        print("  Laplacian eigenvalues: " + " ".join(f"{v:.3f}" for v in vals[: MAX_K + 1]))
        for k in sorted(set(ks.values())):
            lab = partition(V, k)
            print(f"  k={k}:")
            for i in range(k):
                grp = [x for x, l in zip(nbrs, lab) if l == i]
                print(f"    [{len(grp):2d}] {', '.join(grp[:nn])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--topk", default="40,100")
    ap.add_argument("--max-pairs", type=int, default=1500)
    ap.add_argument("--max-words", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-r", type=int, default=20, help="control pools per word (null_calibrated)")
    ap.add_argument("--stab-b", type=int, default=8, help="subsample pairs per k (stability)")
    ap.add_argument("--demo", default=None)
    ap.add_argument("--no-task", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    words, R = load_raw(args.embedding, args.max_words)
    emb = Emb(words, _unit_rows(R))
    print(f"loaded {len(words)} words x {R.shape[1]}d from {args.embedding}")
    if args.demo:
        demo(emb, args.demo, int(args.topk.split(",")[0]), args.seed, args.null_r, args.stab_b)
    if args.no_task:
        return 0

    import nltk
    from nltk.corpus import wordnet as wn
    try:
        wn.synsets("chien", lang="fra")
    except LookupError:
        for r in ("wordnet", "omw-1.4"):
            nltk.download(r, quiet=True)

    np.random.seed(args.seed)
    cache, rows = {}, []
    for tk in [int(t) for t in args.topk.split(",")]:
        recs = collect(emb, wn, tk, args.max_pairs, cache, args.seed, args.null_r, args.stab_b)
        report(recs, f"K={tk}")
        for r in recs:
            r["K"] = tk
        rows.extend(recs)
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
