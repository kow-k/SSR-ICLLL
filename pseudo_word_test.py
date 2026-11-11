#!/usr/bin/env python3
"""
pseudo_word_test.py -- does SSR separate senses when the senses are known exactly?

Pseudo-words (Schutze 1998 style, at the vector level): two unrelated monosemous
nouns w1, w2 are merged into one planted "word"
        p = normalize( alpha * u1 + (1 - alpha) * u2 )        (u = unit vectors)
so p has exactly two senses, with w1 dominant for alpha > 0.5.

SSR on p, exactly as on a real word: the K nearest neighbors of p (w1 and w2
themselves excluded) are clustered spectrally into k = 2 groups.

Scoring, with no sense inventory:
  truth   each pool word is labeled by the source it is closer to (cos to u1 vs u2)
  ARI     adjusted Rand index between SSR's split and the truth labels
          (chance-corrected: a random split scores ~0, a perfect split 1);
          a random partition of the same pool is reported alongside
  minor   share of the pool labeled with the minor source (how much of the minor
          sense reaches the pool at all)
  found   a cluster for the minor sense exists: some cluster's centroid is closer
          to the minor source than to the major one
  both    both clusters are recovered: the two centroids are closer to different sources
Reported per alpha in {0.5, 0.6, 0.7, 0.8, 0.9} and per K.

Optional --kstats: the stability rule of adaptive_k chooses k for the pseudo-word
and for its dominant source word alone; a good rule says k >= 2 for p and k = 1
for the (monosemous) source.

Caveat: a linear mixture assumes the superposition model, so this tests the
separation mechanism, not whether real polysemous vectors are such mixtures.

Run from the repository root (needs sense_explorer/, io_utils.py, adaptive_k.py,
NLTK wordnet for choosing monosemous nouns):
  python pseudo_word_test.py --embedding glove.6B.100d.txt --pairs 200 --topk 50,100 --csv pw_100d.csv --show 5
"""
import argparse
import csv
import sys

import numpy as np

from io_utils import Emb, _unit_rows
from adaptive_k import load_raw, partition, choose_all
from mono_vs_poly_k import strict_mono
from adaptive_k import ari

ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-10)


def pick_pairs(emb, wn, n_pairs, lo, hi, max_cos, rng):
    lemmas = set(wn.all_lemma_names())
    pool = [i for i in range(lo, min(hi, len(emb.words)))
            if (w := emb.words[i]).isalpha() and w.islower() and len(w) >= 4
            and w in lemmas and strict_mono(w, wn)]
    nb = {}
    def nbrs(i):
        if i not in nb:
            nb[i] = set(emb.neighbors(emb.words[i], 50))
        return nb[i]
    pairs, used, tries = [], set(), 0
    while len(pairs) < n_pairs and tries < 50 * n_pairs:
        tries += 1
        a, b = rng.choice(pool, 2, replace=False)
        if a in used or b in used:
            continue
        if float(emb.M[a] @ emb.M[b]) > max_cos:
            continue
        wa, wb = emb.words[a], emb.words[b]
        if nbrs(a) & nbrs(b) or wa in nbrs(b) or wb in nbrs(a):
            continue
        pairs.append((wa, wb)); used |= {a, b}
    return pairs, len(pool)


def pool_of(emb, p, K, exclude):
    sims = emb.M @ p
    for w in exclude:
        sims[emb.index[w]] = -np.inf
    idx = np.argpartition(-sims, K)[:K]
    idx = idx[np.argsort(-sims[idx])]
    return [emb.words[j] for j in idx]


def evaluate(emb, w1, w2, alpha, K, rng):
    u1, u2 = emb.M[emb.index[w1]], emb.M[emb.index[w2]]
    p = unit(alpha * u1 + (1 - alpha) * u2)
    pool = pool_of(emb, p, K, (w1, w2))
    V = np.array([emb.M[emb.index[x]] for x in pool])
    truth = (V @ u2 > V @ u1).astype(int)            # 0 = w1 (major for alpha>.5), 1 = w2
    lab = partition(V, 2)
    rnd = rng.integers(0, 2, len(pool))
    minor = 1 if alpha >= 0.5 else 0
    major_u, minor_u = (u1, u2) if minor == 1 else (u2, u1)
    cents = [unit(V[lab == i].mean(0)) for i in range(2) if np.any(lab == i)]
    closer_minor = [float(c @ minor_u) > float(c @ major_u) for c in cents]
    return dict(word1=w1, word2=w2, alpha=alpha, K=K,
                ari=ari(lab, truth) if len(set(truth)) > 1 else np.nan,
                ari_random=ari(rnd, truth) if len(set(truth)) > 1 else np.nan,
                minor_share=float(np.mean(truth == minor)),
                found=bool(any(closer_minor)),
                both=bool(len(cents) == 2 and closer_minor[0] != closer_minor[1])), V, pool, lab


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--topk", default="50,100")
    ap.add_argument("--rank-lo", type=int, default=1000, help="source words from this frequency rank ...")
    ap.add_argument("--rank-hi", type=int, default=30000, help="... up to this rank")
    ap.add_argument("--max-cos", type=float, default=0.15, help="sources must be at most this similar")
    ap.add_argument("--max-words", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kstats", action="store_true", help="also run the stability k rule (slower)")
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
    np.random.seed(args.seed)
    pairs, npool = pick_pairs(emb, wn, args.pairs, args.rank_lo, args.rank_hi, args.max_cos, rng)
    print(f"loaded {len(words)} words x {Rm.shape[1]}d from {args.embedding}")
    print(f"{npool} monosemous source candidates; {len(pairs)} unrelated pairs "
          f"(cos <= {args.max_cos}, no shared top-50 neighbors)")

    rows = []
    for K in [int(t) for t in args.topk.split(",")]:
        print(f"\n[K={K}]")
        print(f"  {'alpha':>5s} {'minor share':>11s} {'ARI':>6s} {'ARI random':>10s} "
              f"{'minor found':>11s} {'both found':>10s}" + ("   k>=2 for p   k>=2 for source" if args.kstats else ""))
        for a in ALPHAS:
            R, kp, ks = [], [], []
            for w1, w2 in pairs:
                r, V, pool, lab = evaluate(emb, w1, w2, a, K, rng)
                if args.kstats:
                    r["k_pseudo"] = choose_all(V, rng, 20, 8)["stability"]
                    Vs = np.array([emb.M[emb.index[x]] for x in emb.neighbors(w1, K)])
                    r["k_source"] = choose_all(Vs, rng, 20, 8)["stability"]
                R.append(r)
            g = lambda k: np.array([r[k] for r in R], float)
            line = (f"  {a:5.1f} {np.nanmean(g('minor_share')):11.3f} {np.nanmean(g('ari')):6.3f} "
                    f"{np.nanmean(g('ari_random')):10.3f} {np.mean(g('found')):11.2f} {np.mean(g('both')):10.2f}")
            if args.kstats:
                line += f"   {np.mean(g('k_pseudo') >= 2):11.2f}   {np.mean(g('k_source') >= 2):15.2f}"
            print(line)
            rows.extend(R)
        if args.show:
            for w1, w2 in pairs[: args.show]:
                for a in (0.5, 0.8):
                    r, V, pool, lab = evaluate(emb, w1, w2, a, K, rng)
                    grp = [", ".join([x for x, l in zip(pool, lab) if l == i][:8]) for i in range(2)]
                    print(f"    {w1}+{w2} alpha={a}: ARI={r['ari']:.2f}  [{grp[0]}]  |  [{grp[1]}]")
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
