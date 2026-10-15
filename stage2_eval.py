#!/usr/bin/env python3
"""stage2_eval.py -- Stage 2 with paper-ready significance. Records per-pair
outcomes so the paired test is exact: SSR-spectral and the random-partition
null are scored on the SAME pairs -> McNemar on discordant pairs. Adds a
bootstrap 95% CI on the lift and a top_k sweep.
  python3 stage2_eval.py --embedding glove-wiki-gigaword-200.gz
  python3 stage2_eval.py --embedding glove-wiki-gigaword-100.gz --topk 60
  python3 stage2_eval.py --selftest"""
import argparse
import numpy as np
from stage1_anchor import members, best_translation, pick_M, POS
from stage2_score_emb import assign_and_score_emb

MIN_NEIGHBORS = 8; SEED = 0

def collect_records(emb, wn, top_k, max_pairs, cache):
    from sense_explorer.spectral import spectral_clustering
    import random as _r
    pyrng = _r.Random(SEED); nprng = np.random.default_rng(SEED)
    names = sorted({l for l in wn.all_lemma_names(pos=POS) if l.isalpha() and len(l) >= 3})
    pyrng.shuffle(names); recs = []; n = 0
    for x in names:
        if x not in emb.index: continue
        S = wn.synsets(x, pos=POS)
        if len(S) < 2: continue
        tr = best_translation(x, wn)
        if tr is None: continue
        y, shared, S = tr
        M = pick_M(shared); Ds = len(shared) / len(S); k = len(S)
        for s in S: cache.setdefault(s.name(), members(s))
        nbrs = emb.neighbors(x, top_k)
        if len(nbrs) < MIN_NEIGHBORS: continue
        V = np.array([emb.M[emb.index[w]] for w in nbrs])
        lab, _ = spectral_clustering(V, k=k)
        cl_s = [set() for _ in range(k)]
        for w, l in zip(nbrs, lab): cl_s[int(l)].add(w.lower())
        cl_s = [c for c in cl_s if c]
        rl = nprng.integers(0, k, len(nbrs)); cl_r = [set() for _ in range(k)]
        for w, l in zip(nbrs, rl): cl_r[int(l)].add(w.lower())
        cl_r = [c for c in cl_r if c]
        if len(cl_s) < 2 or len(cl_r) < 2: continue
        rs, _ = assign_and_score_emb(cl_s, S, M, emb, cache)
        rr, _ = assign_and_score_emb(cl_r, S, M, emb, cache)
        recs.append((bool(rs), bool(rr), Ds)); n += 1
        if max_pairs and n >= max_pairs: break
    return recs

def mcnemar(spec, null):
    from scipy.stats import binomtest
    b = int(np.sum(spec & ~null)); c = int(np.sum(~spec & null))
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, p

def boot_ci(spec, null, n=5000, rng=None):
    rng = rng or np.random.default_rng(0)
    m = len(spec); idx = rng.integers(0, m, (n, m))
    lifts = spec[idx].mean(1) - null[idx].mean(1)
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))

def stats(recs, label=""):
    spec = np.array([r[0] for r in recs]); null = np.array([r[1] for r in recs])
    Ds = np.array([r[2] for r in recs])
    lift = spec.mean() - null.mean(); ci = boot_ci(spec, null); b, c, p = mcnemar(spec, null)
    print(f"\n[{label}]  n={len(recs)}")
    print(f"  SSR={spec.mean():.3f}  null={null.mean():.3f}  lift={lift:+.3f}  "
          f"95%CI[{ci[0]:+.3f},{ci[1]:+.3f}]")
    print(f"  McNemar: SSR-only-correct={b}  null-only-correct={c}  p={p:.2e}")
    for t, m in [("low ", Ds < 0.5), ("mid ", (Ds >= 0.5) & (Ds < 0.9)), ("high", Ds >= 0.9)]:
        if m.sum() >= 5:
            print(f"    D_s {t}: lift={spec[m].mean()-null[m].mean():+.3f}  (n={int(m.sum())})")

def selftest():
    rng = np.random.default_rng(0); n = 1500
    spec = rng.random(n) < 0.85; null = rng.random(n) < 0.45
    stats([(bool(a), bool(b), float(d)) for a, b, d in zip(spec, null, rng.random(n))],
          "selftest structured")
    spec2 = rng.random(n) < 0.50; null2 = rng.random(n) < 0.50
    stats([(bool(a), bool(b), float(d)) for a, b, d in zip(spec2, null2, rng.random(n))],
          "selftest null")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding"); ap.add_argument("--topk", default="40,60,100")
    ap.add_argument("--max-pairs", type=int, default=1500); ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest: selftest(); return
    if not args.embedding: ap.error("give --embedding PATH or --selftest")
    import nltk
    for r in ("wordnet", "omw-1.4"): nltk.download(r, quiet=True)
    from nltk.corpus import wordnet as wn
    wn.synsets("dog")
    from io_utils import load_glove
    emb = load_glove(args.embedding, max_words=60000)
    print(f"loaded {len(emb.words)} words; embedding={args.embedding}")
    cache = {}
    for tk in [int(t) for t in args.topk.split(",")]:
        stats(collect_records(emb, wn, tk, args.max_pairs, cache), label=f"top_k={tk}")

if __name__ == "__main__":
    main()