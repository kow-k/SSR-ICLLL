#!/usr/bin/env python3
"""
mono_vs_poly_k.py -- can a k rule tell a one-sense word from a many-sense word?

Two probe sets, drawn from the same embedding vocabulary and matched on frequency:
  mono   strict monosemous: a base-form noun with exactly one WordNet synset across
         ALL parts of speech, not a proper name, attested in SemCor counts
  poly   homonyms: >= 2 noun synsets, and the two most frequent noun synsets lie in
         different WordNet lexicographer files (e.g. noun.animal vs noun.artifact)
         (--poly any: simply >= 2 synsets of any part of speech)
  poly from --poly-file: a curated list with a polysemy_type column (homonymy,
         regular_polysemy, metaphor); results are reported per type.
Frequency matching: WordNet sets are sampled equally from log-spaced rank bands; with
--poly-file, each curated word is paired with the mono word nearest in rank.

For every probe word, the K nearest neighbors are built exactly as in adaptive_k.py
and each k rule (same code) chooses k. A good rule says k = 1 for mono words and
k > 1 for poly words. Reported per rule and per K:
  P(k=1 | mono), P(k>1 | poly), balanced accuracy, mean k in each set, and
  AUC of a continuous "split score" (no threshold; 0.5 = cannot tell the sets apart):
    eigengap_current / eigengap_k1   largest gap among k>=2 minus the first gap
    fiedler                          1 - lambda_1 (strength of the best 2-way cut)
    null_calibrated                  how far lambda_1 lies below the control pools (z)
    stability                        best stability margin over the control (in SEs)

  --show N   prints N mono words that the stability rule split, with their groups,
             to see what kind of structure is being taken for senses.

Run from the repository root (needs sense_explorer/, io_utils.py, adaptive_k.py,
NLTK wordnet):
  python mono_vs_poly_k.py --embedding glove.6B.100d.txt --topk 40,100 --n 400 --csv mvp_100d.csv --show 10
  python mono_vs_poly_k.py --embedding glove.6B.100d.txt --topk 40,100 \
      --poly-file data/polysemous_300words.csv --csv mvpc_100d.csv --show 10
"""
import argparse
import csv
import sys

import numpy as np
from scipy.stats import rankdata

from io_utils import Emb, _unit_rows
from adaptive_k import (MAX_K, spectrum, control_pool, stability_scores, partition,
                        find_k_by_eigengap, load_raw)

RULES = ["eigengap_current", "eigengap_k1", "null_calibrated", "stability"]
SCORES = ["eigengap", "fiedler", "null_calibrated", "stability"]


# ---------------------------------------------------------------- probe sets
def _count(s, w):
    return sum(l.count() for l in s.lemmas() if l.name().lower() == w)


def strict_mono(w, wn):
    """Base-form noun with exactly one synset in any POS, not a name, attested in SemCor."""
    if wn.morphy(w, "n") != w:
        return False
    S = wn.synsets(w)
    return (len(S) == 1 and S[0].pos() == "n" and not S[0].instance_hypernyms()
            and _count(S[0], w) >= 1)


def build_sets_curated(words, wn, poly_file, types, rng):
    """poly = curated words of the given types; each matched 1:1 to the strict-mono
    word nearest in frequency rank (without replacement)."""
    rank = {w: i for i, w in enumerate(words)}
    poly, seen = [], set()
    with open(poly_file, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            w = (r.get("word") or "").strip().lower()
            t = (r.get("polysemy_type") or "").strip()
            if not w or w.startswith("#") or w in seen or w not in rank or t not in types:
                continue
            seen.add(w); poly.append((rank[w], w, t))
    lemmas = set(wn.all_lemma_names())
    mono_pool = [(i, w) for i, w in enumerate(words)
                 if w.isalpha() and w.islower() and len(w) >= 3 and w in lemmas and strict_mono(w, wn)]
    used, mono = set(), []
    for rk, w, t in sorted(poly):
        cand = min((abs(i - rk), i, m) for i, m in mono_pool if m not in used)
        used.add(cand[2]); mono.append((cand[1], cand[2], t))
    return mono, poly, len(mono_pool), len(poly)


def build_sets(words, wn, mode, n, n_bands, rng):
    lemmas = set(wn.all_lemma_names())
    mono, poly = [], []
    for rank, w in enumerate(words):
        if not (w.isalpha() and w.islower() and len(w) >= 3 and w in lemmas):
            continue
        allS = wn.synsets(w)
        if len(allS) == 1:
            if strict_mono(w, wn):
                mono.append((rank, w))
            continue
        if mode == "any":
            if len(allS) >= 2:
                poly.append((rank, w))
            continue
        N = wn.synsets(w, pos="n")
        if len(N) >= 2 and N[0].lexname() != N[1].lexname():
            poly.append((rank, w))
    edges = np.unique(np.geomspace(1, len(words) + 1, n_bands + 1).astype(int))
    per_band = max(1, n // (len(edges) - 1))
    out_m, out_p = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bm = [x for x in mono if lo - 1 <= x[0] < hi - 1]
        bp = [x for x in poly if lo - 1 <= x[0] < hi - 1]
        m = min(len(bm), len(bp), per_band)
        if m == 0:
            continue
        out_m += [bm[i] for i in rng.choice(len(bm), m, replace=False)]
        out_p += [bp[i] for i in rng.choice(len(bp), m, replace=False)]
    return [x + ("wordnet",) for x in out_m], [x + ("wordnet",) for x in out_p], len(mono), len(poly)


# ---------------------------------------------------------------- rules + scores
def analyse(V, rng, R, B):
    vals, _ = spectrum(V, MAX_K + 2)
    gaps = np.diff(vals[: MAX_K + 1])
    out = {"eigengap_current": find_k_by_eigengap(vals, 2, 5),
           "eigengap_k1": int(np.argmax(gaps[:MAX_K])) + 1}
    sc = {"eigengap": float(np.max(gaps[1:MAX_K]) - gaps[0]),
          "fiedler": float(1 - vals[1])}
    # null-calibrated
    null = np.array([spectrum(control_pool(V, rng), MAX_K + 1)[0] for _ in range(R)])
    q = np.percentile(null, 5, axis=0)
    k = 1
    for i in range(1, min(MAX_K, len(vals))):
        if vals[i] < q[i]:
            k += 1
        else:
            break
    out["null_calibrated"] = k
    sc["null_calibrated"] = float((null[:, 1].mean() - vals[1]) / (null[:, 1].std(ddof=1) + 1e-12))
    # stability
    real = stability_scores(V, rng, B)
    ctrl = stability_scores(control_pool(V, rng), rng, B)
    best, best_diff, best_t = 1, 0.0, -np.inf
    for kk in range(2, MAX_K + 1):
        d = real[kk].mean() - ctrl[kk].mean()
        se = np.sqrt(real[kk].var(ddof=1) / B + ctrl[kk].var(ddof=1) / B) + 1e-12
        best_t = max(best_t, d / se)
        if d > 2 * se and d > best_diff:
            best, best_diff = kk, d
    out["stability"] = best
    sc["stability"] = float(best_t)
    return out, sc


def auc(pos, neg):
    """P(score of a poly word > score of a mono word); ties count 1/2."""
    x = np.r_[pos, neg]
    r = rankdata(x)
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def report(recs, K):
    types = sorted({r["type"] for r in recs})
    if len(types) > 1:
        for t in types:
            report_one([r for r in recs if r["type"] == t], f"{K}, {t}")
        report_one(recs, f"{K}, all types")
    else:
        report_one(recs, K)


def report_one(recs, K):
    m = [r for r in recs if r["set"] == "mono"]; p = [r for r in recs if r["set"] == "poly"]
    rk = lambda S: np.median([r["rank"] for r in S])
    print(f"\n[K={K}]  mono n={len(m)} (median rank {rk(m):.0f})   poly n={len(p)} (median rank {rk(p):.0f})")
    print(f"  {'rule':18s} {'P(k=1|mono)':>11s} {'P(k>1|poly)':>11s} {'bal.acc':>8s} "
          f"{'mean k mono':>11s} {'mean k poly':>11s}")
    for name in RULES:
        km = np.array([r[f"{name}_k"] for r in m]); kp = np.array([r[f"{name}_k"] for r in p])
        a, b = np.mean(km == 1), np.mean(kp > 1)
        print(f"  {name:18s} {a:11.2f} {b:11.2f} {(a+b)/2:8.2f} {km.mean():11.2f} {kp.mean():11.2f}")
    print(f"  {'split score':18s} {'AUC (poly > mono)':>20s} {'95% CI':>16s}")
    rng = np.random.default_rng(0)
    for s in SCORES:
        ps = np.array([r[f's_{s}'] for r in p]); ms = np.array([r[f's_{s}'] for r in m])
        bs = [auc(ps[rng.integers(0, len(ps), len(ps))], ms[rng.integers(0, len(ms), len(ms))]) for _ in range(1000)]
        print(f"  {s:18s} {auc(ps, ms):20.3f}   [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")


def show(emb, recs, K, n, nn=7):
    split = [r for r in recs if r["set"] == "mono" and r["stability_k"] > 1][:n]
    if not split:
        return
    print(f"\n  mono words that the stability rule split (K={K}):")
    np.random.seed(0)
    for r in split:
        nbrs = emb.neighbors(r["word"], K)
        V = np.array([emb.M[emb.index[x]] for x in nbrs])
        lab = partition(V, r["stability_k"])
        groups = [", ".join([x for x, l in zip(nbrs, lab) if l == i][:nn]) for i in range(r["stability_k"])]
        print(f"    {r['word']} ({r['gloss'][:50]}):  " + "  |  ".join(f"[{g}]" for g in groups))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--topk", default="40,100")
    ap.add_argument("--n", type=int, default=400, help="target words per set")
    ap.add_argument("--bands", type=int, default=8, help="frequency bands for matching")
    ap.add_argument("--poly", choices=["homonym", "any"], default="homonym")
    ap.add_argument("--poly-file", default=None,
                    help="curated list (word, polysemy_type), e.g. data/polysemous_300words.csv")
    ap.add_argument("--types", default="homonymy,regular_polysemy,metaphor",
                    help="polysemy_type values to use from --poly-file")
    ap.add_argument("--max-words", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-r", type=int, default=20)
    ap.add_argument("--stab-b", type=int, default=8)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    from nltk.corpus import wordnet as wn
    try:
        wn.synsets("dog")
    except LookupError:
        import nltk; nltk.download("wordnet", quiet=True)

    words, Rm = load_raw(args.embedding, args.max_words)
    emb = Emb(words, _unit_rows(Rm))
    rng = np.random.default_rng(args.seed)
    if args.poly_file:
        mono, poly, nm, npl = build_sets_curated(words, wn, args.poly_file, set(args.types.split(",")), rng)
    else:
        mono, poly, nm, npl = build_sets(words, wn, args.poly, args.n, args.bands, rng)
    print(f"loaded {len(words)} words x {Rm.shape[1]}d from {args.embedding}")
    print(f"candidates: {nm} monosemous, {npl} {args.poly} polysemous; "
          f"frequency-matched sample: {len(mono)} + {len(poly)}")

    rows = []
    for K in [int(t) for t in args.topk.split(",")]:
        recs = []
        for tag, S in (("mono", mono), ("poly", poly)):
            for j, (rank, w, ptype) in enumerate(S):
                nbrs = emb.neighbors(w, K)
                V = np.array([emb.M[emb.index[x]] for x in nbrs])
                saved = np.random.get_state()
                np.random.seed((args.seed * 1000003 + K * 7919 + rank) % (2**31))
                try:
                    ks, sc = analyse(V, np.random.default_rng(args.seed + rank), args.null_r, args.stab_b)
                finally:
                    np.random.set_state(saved)
                rec = dict(K=K, set=tag, type=ptype, word=w, rank=rank, n_synsets=len(wn.synsets(w)),
                           gloss=wn.synsets(w)[0].definition())
                rec.update({f"{k}_k": v for k, v in ks.items()})
                rec.update({f"s_{k}": v for k, v in sc.items()})
                recs.append(rec)
            print(f"  K={K}: {tag} done", file=sys.stderr)
        report(recs, K)
        if args.show:
            show(emb, recs, K, args.show)
        rows.extend(recs)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
