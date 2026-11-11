#!/usr/bin/env python3
"""
coherence_check.py -- Section IV-D (cross-word coherence, Fig. 2): does it show more
than plain word vectors would?

Paper's claims: (a) pooled sense vectors group by semantic domain; (b) a word whose
senses lie in different domains has its sense vectors land in those different domains.

Domains: domain_primary of data/polysemous_300words.csv (9 domains). A word listed in
several rows (e.g. mouse / mouse_technology) gets the set of their domains D(w).

Reference space: the plain word vectors (one per base word) of all list words. Every
query vector is judged by its m nearest reference words, the query's own word excluded.

Query conditions (one vector set per word):
  baseline   the word's own vector (one vector)
  senses     the sense vectors in --senses (Fig. 2 data, or a centroid rerun)
  random     centroids of a random partition of the word's top_k neighbors into as many
             groups as the word has sense vectors (the lenient null)
Metrics:
  A  domain coherence: fraction of a vector's m nearest reference words that share a
     domain with its word. Swap null: the same neighbors scored against the domains of
     another word with the same number of domains.
  B  split coverage (words with >= 2 domains): each vector gets the majority domain of
     its m nearest reference words; success iff the word's vectors cover all its domains.
     Swap null: coverage of another word's domain set. The baseline (one vector) cannot
     cover two domains, so B is the claim specific to sense vectors.
Bootstrap 95% CIs over words.

Run from the repository root:
  python coherence_check.py --embedding glove.6B.100d.txt \
      --senses data/polysemous_300words-glove-wiki-100d.json --words data/polysemous_300words.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from io_utils import Emb, _unit_rows
from adaptive_k import load_raw


def unit(v):
    return v / (np.linalg.norm(v) + 1e-10)


def load_domains(path):
    D, order = defaultdict(set), []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            w = (r.get("word") or "").strip().lower()
            if not w or w.startswith("#"):
                continue
            D[w].add(r["domain_primary"].strip().lower())
            if w not in order:
                order.append(w)
    return D, order


def load_senses(path, D):
    raw = json.load(open(path))
    out = {}
    for key, senses in raw.items():
        base = key if key in D else key.split("_")[0]
        if base not in D or base in out:        # duplicate rows hold identical vectors
            continue
        vecs = [unit(np.asarray(v, float)) for k, v in senses.items() if k != "default"]
        if vecs:
            out[base] = vecs
    return out


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--senses", required=True, help="sense-vector JSON (Fig. 2 data or a centroid rerun)")
    ap.add_argument("--words", default="data/polysemous_300words.csv")
    ap.add_argument("--max-words", type=int, default=400000)
    ap.add_argument("--m", type=int, default=10, help="reference neighbors per query vector")
    ap.add_argument("--top-k", type=int, default=50, help="neighbor pool for the random-partition null")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    D, order = load_domains(args.words)
    words, Rm = load_raw(args.embedding, args.max_words)
    emb = Emb(words, _unit_rows(Rm))
    S = load_senses(args.senses, D)
    W = [w for w in order if w in emb.index and w in S]
    print(f"{len(D)} list words, {len(W)} with sense vectors and in the embedding; "
          f"{sum(len(D[w]) >= 2 for w in W)} have >= 2 domains "
          f"({', '.join(w + ':' + '/'.join(sorted(D[w])) for w in W if len(D[w]) >= 2)})")
    ref = np.array([emb.M[emb.index[w]] for w in W])
    ref_dom = [D[w] for w in W]
    widx = {w: i for i, w in enumerate(W)}

    def nbr_domains(v, w):
        sims = ref @ v
        sims[widx[w]] = -np.inf
        top = np.argsort(-sims)[: args.m]
        return [ref_dom[j] for j in top]

    cond = {"baseline": {w: [emb.M[emb.index[w]]] for w in W}, "senses": {w: S[w] for w in W}, "random": {}}
    for w in W:
        k = len(S[w])
        nb = emb.neighbors(w, args.top_k)
        V = np.array([emb.M[emb.index[x]] for x in nb])
        lab = rng.integers(0, k, len(nb)) if k > 1 else np.zeros(len(nb), int)
        cond["random"][w] = [unit(V[lab == i].mean(0)) for i in range(k) if np.any(lab == i)]

    # swap partners: same number of domains, no domain in common if possible
    partner = {}
    for w in W:
        cand = [y for y in W if y != w and len(D[y]) == len(D[w]) and not (D[y] & D[w])] or \
               [y for y in W if y != w and len(D[y]) == len(D[w])] or \
               [y for y in W if y != w and not (D[y] & D[w])] or [y for y in W if y != w]
        partner[w] = cand[rng.integers(len(cand))]

    print(f"\nA  domain coherence (share of {args.m} nearest list words sharing a domain); 95% CI over words")
    for c in ("baseline", "senses", "random"):
        real, swap = [], []
        for w in W:
            nd = [nbr_domains(v, w) for v in cond[c][w]]
            real.append(np.mean([np.mean([bool(d & D[w]) for d in ds]) for ds in nd]))
            swap.append(np.mean([np.mean([bool(d & D[partner[w]]) for d in ds]) for ds in nd]))
        real, swap = np.array(real), np.array(swap)
        lo, hi = boot(real - swap)
        print(f"  {c:9s} {real.mean():.3f}   swap null {swap.mean():.3f}   real - swap {real.mean()-swap.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    multi = [w for w in W if len(D[w]) >= 2]
    print(f"\nB  split coverage, {len(multi)} words with >= 2 domains (all domains covered by the word's vectors' majority domains)")
    res = {}
    for c in ("baseline", "senses", "random"):
        real, swap = [], []
        for w in multi:
            cov = set()
            for v in cond[c][w]:
                cnt = Counter(d for ds in nbr_domains(v, w) for d in ds)
                if cnt:
                    cov.add(cnt.most_common(1)[0][0])
            real.append(D[w] <= cov)
            swap.append(D[partner[w]] <= cov)
        real, swap = np.array(real, float), np.array(swap, float)
        res[c] = real
        lo, hi = boot(real - swap)
        nv = np.mean([len(cond[c][w]) for w in multi])
        print(f"  {c:9s} {real.mean():.3f}   swap null {swap.mean():.3f}   real - swap {real.mean()-swap.mean():+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]   (mean vectors/word {nv:.2f})")
    lo, hi = boot(res["senses"] - res["random"])
    print(f"  senses - random  {res['senses'].mean()-res['random'].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    ok = [w for w in multi if res["senses"][multi.index(w)]]
    print(f"\n  words whose sense vectors cover all their domains ({len(ok)}): {', '.join(ok[:40])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
