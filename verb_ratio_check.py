#!/usr/bin/env python3
"""
verb_ratio_check.py -- Section IV-E (BNC verbs): do the reported numbers distinguish
SSR from any vectors built out of a verb's own neighborhood?

Paper's claims (from within_between_metrics.py, cosine distance):
  * a polysemous verb's sense vectors are closer to each other than to other verbs
    (within/between ratio ~0.30: "local refinement");
  * between-word distance shrinks ~40% relative to the baseline (one vector per verb).
Both could hold for ANY set of vectors derived from a verb's neighborhood, because such
vectors share the neighborhood and centroids are pulled toward the common mean.

Conditions, all on the same verbs (those with >= 2 sense vectors in --senses), each verb
with the same number of vectors n_v as in --senses:
  senses     the sense vectors in --senses (paper: data/bnc_verbs_120_ssr.json)
  spectral   fresh spectral clusters of the verb's top_k neighbors into n_v groups, centroids
  random     a random partition of the same neighbors into n_v groups, centroids
  halves     centroids of n_v random SUBSETS (half the pool each, overlapping allowed):
             near-duplicates of the pool mean, i.e. "no separation at all"
  baseline   the verb's own vector (d_between only)
Reported: mean d_within, d_between, ratio (polysemous verbs), with bootstrap 95% CIs
for senses - random. If random/halves reproduce the paper's pattern, IV-E's numbers are
not evidence of sense refinement.

Run from the repository root (BNC embedding; tokens like run_VERB):
  python verb_ratio_check.py --embedding /path/to/bnc/model.txt \
      --verbs data/bnc_verbs_120.csv --senses data/bnc_verbs_120_ssr.json
"""
import argparse
import csv
import json
import sys

import numpy as np

from io_utils import Emb, _unit_rows
from adaptive_k import load_raw, partition


def unit(v):
    return v / (np.linalg.norm(v) + 1e-10)


def cosd(A, B):
    return 1.0 - unit_rows(A) @ unit_rows(B).T


def unit_rows(A):
    A = np.atleast_2d(np.asarray(A, float))
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)


def metrics(groups):
    """groups: {verb: (n,d) array}. Returns per-verb within, between, ratio (as in within_between_metrics)."""
    names = list(groups)
    out = {}
    for w in names:
        G = groups[w]
        others = np.vstack([groups[x] for x in names if x != w])
        Dw = cosd(G, G)
        iu = np.triu_indices(len(G), 1)
        dw = float(Dw[iu].mean()) if len(G) > 1 else 0.0
        db = float(cosd(G, others).mean())
        out[w] = (dw, db, dw / db if db > 0 else np.nan)
    return out


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--verbs", default="data/bnc_verbs_120.csv")
    ap.add_argument("--senses", default="data/bnc_verbs_120_ssr.json")
    ap.add_argument("--max-words", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    tok = {}
    with open(args.verbs, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("verb") or r.get("word") or "").strip()
            if v and not v.startswith("#"):
                tok[v] = (r.get("bnc_token") or v).strip()
    raw = json.load(open(args.senses))
    senses = {}
    for v, s in raw.items():
        vecs = [np.asarray(x, float) for k, x in s.items() if k != "default"]
        if len(vecs) >= 2 and v in tok:
            senses[v] = np.array(vecs)
    words, Rm = load_raw(args.embedding, args.max_words or 10**9)
    emb = Emb(words, _unit_rows(Rm))
    verbs = [v for v in senses if tok[v] in emb.index]
    print(f"{len(senses)} verbs with >= 2 sense vectors; {len(verbs)} found in the embedding")

    cond = {c: {} for c in ("senses", "spectral", "random", "halves")}
    base = {}
    for v in verbs:
        t = tok[v]; n = len(senses[v])
        nb = emb.neighbors(t, args.top_k)
        V = np.array([emb.M[emb.index[x]] for x in nb])
        cond["senses"][v] = unit_rows(senses[v])
        lab = partition(V, n)
        cond["spectral"][v] = unit_rows([V[lab == i].mean(0) for i in range(n) if np.any(lab == i)])
        rl = rng.integers(0, n, len(nb))
        cond["random"][v] = unit_rows([V[rl == i].mean(0) for i in range(n) if np.any(rl == i)])
        cond["halves"][v] = unit_rows([V[rng.choice(len(V), len(V) // 2, replace=False)].mean(0) for _ in range(n)])
        base[v] = emb.M[emb.index[t]][None, :]

    bm = metrics(base)
    print(f"\nbaseline (one vector per verb): d_between {np.mean([x[1] for x in bm.values()]):.3f}")
    print(f"\n  {'condition':9s} {'d_within':>9s} {'d_between':>10s} {'ratio':>7s}   (means over {len(verbs)} polysemous verbs)")
    M = {}
    for c in cond:
        m = metrics(cond[c]); M[c] = m
        a = np.array(list(m.values()))
        print(f"  {c:9s} {a[:,0].mean():9.3f} {a[:,1].mean():10.3f} {np.nanmean(a[:,2]):7.3f}")
    for c in ("random", "halves", "spectral"):
        d = [M["senses"][v][2] - M[c][v][2] for v in verbs]
        lo, hi = boot(d)
        print(f"  ratio: senses - {c:8s} {np.mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\nReading: the paper's pattern (ratio < 1, d_between below baseline) is evidence of sense refinement")
    print("only if 'senses' differs clearly from 'random' and 'halves', which have no sense structure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
