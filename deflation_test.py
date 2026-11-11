#!/usr/bin/env python3
"""
deflation_test.py (v4) -- recovering minor senses that are NOT balanced, by deflation.

Under superposition e ~ a*s1 + (1-a)*s2, the minor sense is what remains once the
dominant sense is removed. Pipeline for a word vector p (unit):
  1. estimate the dominant sense s1_hat from p's K-neighbor pool:
       --s1 cluster  centroid of the larger of 2 spectral clusters (v1 behavior)
       --s1 pool     centroid of the whole pool (when one sense dominates, the 2 clusters
                     split WITHIN it, so the larger cluster is only a facet)
  2. deflate: r = p - kappa * s1_hat
       projection   kappa = p . s1_hat
       prior a      kappa = a / sqrt(a^2 + (1-a)^2)   (assumed dominance a)
  3. s2_hat = centroid of the K2 nearest neighbors of r
  4. iterate (show-words, --rounds): r <- r - (r . s_hat) s_hat, giving s3_hat, ...
"balanced" = SSR today (smaller stage-1 cluster).

Decoy null (v3): "found" asks whether s2_hat is closer to the minor word B than to the
major word A. After deflation removes A's direction, s2_hat might be closer to almost ANY
unrelated word than to A. So each pair gets --decoys unrelated monosemous words D that
were NOT mixed into p, drawn under the same constraints as B (same frequency band, strictly
monosemous, cos <= --max-cos to both sources). Reported per method:
  found     s2_hat closer to B than to A
  decoy     the same criterion with B replaced by a decoy (mean over decoys) = chance level
  top1      s2_hat closer to B than to every decoy (chance = 1/(decoys+1))
The lift found - decoy is the part of "found" due to the planted minor sense.

Signal check (v3): share of a monosemous word's energy that its pool centroid leaves
unexplained (1 - cos^2), against the minor sense's share of the pseudo-word's energy.

Detection test (v4): can we tell, from the vector alone, that a minor sense EXISTS?
For a word vector p, dominant estimate s1_hat and found sense s2_hat, three statistics:
  align   (unit(r) . s2_hat)^2   how much of the residual lines up with the sense found
  resid   |r|^2                  energy the deflation leaves (for proj: 1 - (p.s1_hat)^2)
  gain    (p . g)^2, g = s2_hat orthogonalized against s1_hat: energy of p that the
          second sense explains beyond the first
Each is computed for every pseudo-word and for its dominant source word alone (strictly
monosemous, same pool size). Reported per alpha and method: AUC (pseudo > monosemous) and
the detection rate at a 5% false-alarm rate (threshold = monosemous 95th percentile).
For --show-words, each statistic's percentile among the monosemous words is printed.

False-sense test (v2, off by default; --ztest to enable): a residual direction counts as a sense only if its neighborhood
is more coherent than random directions at the same angle from p. Random directions are
built from random vocabulary words (orthogonalized against p, then rotated to the
residual's angle), so they live where real words live. z = (coherence - mean_null)/sd_null.
Reported: AUC of z (pseudo-word minor sense vs monosemous word's residual) and the share
passing z > 2 in each group.

Pseudo-words (same pairs/seed as pseudo_word_test.py): p = unit(a*u1 + (1-a)*u2).
  found     s2_hat closer to the true minor word B than to A (A = source closer to s1_hat)
  cos_B     cos(s2_hat, u_B);  B share: share of r's K2 neighbors closer to u_B than u_A

Run from the repository root:
  python deflation_test.py --embedding glove.6B.100d.txt --pairs 200 --topk 50 \
      --show-words bank,bat,spring,crane,mouse,plant,cell --rounds 3 --csv defl2_100d.csv
"""
import argparse
import csv
import sys

import numpy as np
from scipy.stats import rankdata

from io_utils import Emb, _unit_rows
from adaptive_k import load_raw, partition
from pseudo_word_test import pick_pairs, pool_of, unit
from mono_vs_poly_k import strict_mono

ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]


def kappa_prior(a):
    return a / np.sqrt(a * a + (1 - a) ** 2)


def stage1(emb, p, K, exclude):
    pool = pool_of(emb, p, K, exclude)
    V = np.array([emb.M[emb.index[x]] for x in pool])
    lab = partition(V, 2)
    sizes = [np.sum(lab == i) for i in range(2)]
    big = int(np.argmax(sizes))
    return dict(pool=pool, V=V, lab=lab, big=big,
                s1_cluster=unit(V[lab == big].mean(0)),
                s1_pool=unit(V.mean(0)),
                s2_bal=unit(V[lab == 1 - big].mean(0)) if sizes[1 - big] > 0 else None)


def neighbors_of(emb, r, K2, excl_idx):
    sims = emb.M @ r
    sims[excl_idx] = -np.inf
    idx = np.argpartition(-sims, K2)[:K2]
    return idx[np.argsort(-sims[idx])]


def coherence(emb, idx):
    V = emb.M[idx]
    G = V @ V.T
    return float(G[np.triu_indices(len(idx), 1)].mean())


def ztest(emb, p, r, K2, excl_idx, R, rng):
    """z of r's neighbor coherence against R random directions at the same angle from p."""
    r = unit(r)
    c = float(np.clip(r @ p, -1, 1)); s = np.sqrt(max(0.0, 1 - c * c))
    W = emb.M[rng.integers(0, len(emb.words), R)]
    W = W - np.outer(W @ p, p)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-10)
    Q = c * p[None, :] + s * W                       # R x d, each at angle acos(c) from p
    S = emb.M @ Q.T
    S[excl_idx, :] = -np.inf
    null = []
    for j in range(R):
        idx = np.argpartition(-S[:, j], K2)[:K2]
        null.append(coherence(emb, idx))
    real = coherence(emb, neighbors_of(emb, r, K2, excl_idx))
    null = np.array(null)
    return (real - null.mean()) / (null.std(ddof=1) + 1e-10), real


def deflate(emb, p, s1, kappa, K2, excl_idx):
    r = p - kappa * s1
    if np.linalg.norm(r) < 1e-8:
        return None, None, None
    idx = neighbors_of(emb, unit(r), K2, excl_idx)
    return unit(emb.M[idx].mean(0)), idx, r


def detect_stats(p, s1, s2, r):
    if s2 is None or r is None:
        return dict(align=np.nan, resid=np.nan, gain=np.nan)
    g = s2 - float(s2 @ s1) * s1
    gn = np.linalg.norm(g)
    return dict(align=float(unit(r) @ s2) ** 2, resid=float(r @ r),
                gain=float(p @ g / gn) ** 2 if gn > 1e-8 else 0.0)


STATS = ("align", "resid", "gain")


def method_list(priors):
    m = [("cluster", "projection"), ("pool", "projection")]
    for a in priors:
        m += [("cluster", a), ("pool", a)]
    return m


def mname(s1mode, k):
    return f"{s1mode}/{'proj' if k == 'projection' else 'prior' + str(k)}"


def kappa_of(k, p, s1):
    return float(p @ s1) if k == "projection" else kappa_prior(k)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--topk", default="50")
    ap.add_argument("--k2", type=int, default=20)
    ap.add_argument("--priors", default="0.7,0.9")
    ap.add_argument("--null-r", type=int, default=10, help="random directions per false-sense test")
    ap.add_argument("--ztest", action="store_true", help="run the (v2) coherence false-sense test; slow")
    ap.add_argument("--decoys", type=int, default=20, help="decoy words per pair")
    ap.add_argument("--rounds", type=int, default=3, help="deflation rounds in --show-words")
    ap.add_argument("--rank-lo", type=int, default=1000)
    ap.add_argument("--rank-hi", type=int, default=30000)
    ap.add_argument("--max-cos", type=float, default=0.15)
    ap.add_argument("--max-words", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-words", default=None)
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
    pairs, _ = pick_pairs(emb, wn, args.pairs, args.rank_lo, args.rank_hi, args.max_cos, rng)
    drng = np.random.default_rng(args.seed + 1)          # separate stream: pairs stay as in v2
    lemmas = set(wn.all_lemma_names())
    used = {w for pr in pairs for w in pr}
    cand = [i for i in range(args.rank_lo, min(args.rank_hi, len(emb.words)))
            if (w := emb.words[i]).isalpha() and w.islower() and len(w) >= 4 and w not in used
            and w in lemmas and strict_mono(w, wn)]
    cand = np.array(cand)
    decoys = {}
    for w1, w2 in pairs:
        u1, u2 = emb.M[emb.index[w1]], emb.M[emb.index[w2]]
        ok = cand[(emb.M[cand] @ u1 <= args.max_cos) & (emb.M[cand] @ u2 <= args.max_cos)]
        decoys[(w1, w2)] = drng.choice(ok, min(args.decoys, len(ok)), replace=False)
    print(f"{len(cand)} decoy candidates; {args.decoys} decoys per pair")
    priors = [float(x) for x in args.priors.split(",") if x]
    M = method_list(priors)
    names = ["balanced"] + [mname(*m) for m in M]
    print(f"loaded {len(words)} words x {Rm.shape[1]}d from {args.embedding}; {len(pairs)} pseudo-word pairs")

    rows, summary, detsum, mono_by_K = [], [], [], {}
    for K in [int(t) for t in args.topk.split(",")]:
        # monosemous control
        mono = {mname(*m): [] for m in M}
        unexpl = {"pool": [], "cluster": []}
        monostat = {mname(*m): [] for m in M}
        mono_by_K[K] = monostat
        for w1, w2 in pairs:
            u1 = emb.M[emb.index[w1]]; ex = [emb.index[w1]]
            st = stage1(emb, u1, K, (w1,))
            for m in unexpl:
                unexpl[m].append(1 - float(u1 @ st["s1_" + m]) ** 2)
            for s1mode, k in M:
                s1 = st["s1_" + s1mode]
                s2, idx, r = deflate(emb, u1, s1, kappa_of(k, u1, s1), args.k2, ex)
                monostat[mname(s1mode, k)].append(detect_stats(u1, s1, s2, r))
                if not args.ztest:
                    continue
                z, coh = ztest(emb, u1, r, args.k2, ex, args.null_r, rng) if r is not None else (np.nan, np.nan)
                mono[mname(s1mode, k)].append(z)
        print(f"\n[K={K}]  cells: found / decoy / top1   (decoy = chance level of 'found'; top1 chance = {1/(args.decoys+1):.2f})")
        print("  true a " + "".join(f"{n:>20s}" for n in names))
        sigs = []
        for a in ALPHAS:
            acc = {n: [] for n in names}
            zs = {mname(*m): [] for m in M}
            snr = []
            pstat = {mname(*m): [] for m in M}
            for w1, w2 in pairs:
                u1, u2 = emb.M[emb.index[w1]], emb.M[emb.index[w2]]
                p = unit(a * u1 + (1 - a) * u2)
                ex = [emb.index[w1], emb.index[w2]]
                st = stage1(emb, p, K, (w1, w2))
                ref = st["s1_cluster"]
                A, B = (u1, u2) if ref @ u1 >= ref @ u2 else (u2, u1)
                rec = dict(K=K, alpha=a, word1=w1, word2=w2)
                Dm = emb.M[decoys[(w1, w2)]]

                def score(s2, A_, B_):
                    cA, cB, cD = float(s2 @ A_), float(s2 @ B_), Dm @ s2
                    return (cB > cA, float(np.mean(cD > cA)), bool(cB > cD.max()), cB, float(cD.mean()))
                s2b = st["s2_bal"]
                acc["balanced"].append(score(s2b, A, B) if s2b is not None else (False, 0.0, False, np.nan, np.nan))
                for s1mode, k in M:
                    s1 = st["s1_" + s1mode]
                    if s1mode == "pool":                       # targets relative to this s1
                        A2, B2 = (u1, u2) if s1 @ u1 >= s1 @ u2 else (u2, u1)
                    else:
                        A2, B2 = A, B
                    s2, idx, r = deflate(emb, p, s1, kappa_of(k, p, s1), args.k2, ex)
                    n = mname(s1mode, k)
                    ds = detect_stats(p, s1, s2, r)
                    pstat[n].append(ds)
                    for t in STATS:
                        rec[f"{n}_{t}"] = ds[t]
                    if s2 is None:
                        acc[n].append((False, 0.0, False, np.nan, np.nan)); zs[n].append(np.nan); continue
                    acc[n].append(score(s2, A2, B2))
                    if s1mode == "pool" and k == "projection":
                        snr.append(float(unit(r) @ B2) ** 2)
                    rec[f"{n}_found"], rec[f"{n}_decoy"], rec[f"{n}_top1"], rec[f"{n}_cosB"], rec[f"{n}_cosD"] = acc[n][-1]
                    if args.ztest:
                        z, _ = ztest(emb, p, r, args.k2, ex, args.null_r, rng)
                        zs[n].append(z); rec[f"{n}_z"] = z
                rows.append(rec)
            cells, cos = [], []
            for n in names:
                X = np.array(acc[n], float)
                cells.append(f"{np.mean(X[:, 0]):.2f}/{np.mean(X[:, 1]):.2f}/{np.mean(X[:, 2]):.2f}")
                cos.append(f"{np.nanmean(X[:, 3]):.2f} vs {np.nanmean(X[:, 4]):.2f}")
                summary.append(dict(K=K, alpha=a, method=n, found=np.mean(X[:, 0]), decoy=np.mean(X[:, 1]),
                                    top1=np.mean(X[:, 2]), cosB=np.nanmean(X[:, 3]), cosD=np.nanmean(X[:, 4])))
            print(f"  {a:5.1f}  " + "".join(f"{c:>20s}" for c in cells))
            print(f"  {'cosB vs cosD':>12s}" + "".join(f"{c:>20s}" for c in cos)[7:])
            for t in STATS:
                dt = []
                for m in M:
                    n = mname(*m)
                    P = np.array([d[t] for d in pstat[n]], float); Q = np.array([d[t] for d in monostat[n]], float)
                    thr = np.nanpercentile(Q, 95)
                    au, tp = auc(P, Q), float(np.nanmean(P > thr))
                    dt.append(f"{au:.2f} @5%:{tp:.2f}")
                    detsum.append(dict(K=K, alpha=a, method=n, stat=t, auc=au, det_at_5pct=tp))
                print(f"  {'det ' + t:>12s}" + f"{'':>20s}"[7:] + "".join(f"{x:>20s}" for x in dt))
            msh = (1 - a) ** 2 / (a * a + (1 - a) ** 2)
            sigs.append((a, msh, np.mean(snr)))
            if not args.ztest:
                continue
            ztxt = []
            for m in M:
                n = mname(*m)
                P, Q = np.array(zs[n], float), np.array(mono[n], float)
                ztxt.append(f"{auc(P, Q):.2f} {np.nanmean(P > 2):.2f}|{np.nanmean(Q > 2):.2f}")
            print(f"  {'z-test':>5s}  " + f"{'':>20s}" + "".join(f"{x:>20s}" for x in ztxt))
        print("  (det rows: AUC pseudo-word vs its monosemous source; detection rate at 5% false alarms)")
        if args.ztest:
            print("  (z-test cells: AUC pseudo vs monosemous, then share with z>2: pseudo|monosemous)")
        print("\n  signal check")
        for m in ("pool", "cluster"):
            q = np.percentile(unexpl[m], [25, 50, 75])
            print(f"    monosemous words: energy left unexplained by the {m:7s} centroid  "
                  f"median {q[1]:.2f} (IQR {q[0]:.2f}-{q[2]:.2f})")
        for a, msh, sn in sigs:
            print(f"    alpha {a:.1f}: minor sense = {msh:.2f} of the pseudo-word's energy (if u1 _|_ u2); "
                  f"share of the pool/proj residual along B {sn:.2f}")

    if args.show_words:
        K = int(args.topk.split(",")[0])
        for w in args.show_words.split(","):
            if w not in emb.index:
                print(f"\n{w}: not in vocabulary"); continue
            p = emb.M[emb.index[w]]; ex = [emb.index[w]]
            st = stage1(emb, p, K, (w,))
            print(f"\n=== {w} (K={K}) ===")
            print(f"  stage 1, larger cluster : {', '.join([x for x, l in zip(st['pool'], st['lab']) if l == st['big']][:10])}")
            print(f"  stage 1, smaller cluster: {', '.join([x for x, l in zip(st['pool'], st['lab']) if l != st['big']][:10])}")
            for s1mode, k in [("cluster", priors[-1] if priors else "projection"), ("pool", "projection"),
                              ("pool", priors[-1] if priors else "projection")]:
                s1 = st["s1_" + s1mode]
                r = p - kappa_of(k, p, s1) * s1
                n = mname(s1mode, k)
                s2_, _, r_ = deflate(emb, p, s1, kappa_of(k, p, s1), args.k2, ex)
                ds = detect_stats(p, s1, s2_, r_)
                pct = []
                for t in STATS:
                    Q = np.array([d[t] for d in mono_by_K.get(K, {}).get(n, [])], float)
                    pct.append(f"{t} {ds[t]:.2f} (pct {100 * np.nanmean(Q < ds[t]):.0f})" if len(Q) else f"{t} {ds[t]:.2f}")
                print(f"  [{n}]  detection: " + ", ".join(pct))
                for t in range(1, args.rounds + 1):
                    if np.linalg.norm(r) < 1e-8:
                        break
                    idx = neighbors_of(emb, unit(r), args.k2, ex)
                    zt = ""
                    if args.ztest:
                        z, coh = ztest(emb, p, r, args.k2, ex, args.null_r, rng); zt = f"z={z:+.1f}, "
                    s = unit(emb.M[idx].mean(0))
                    print(f"    round {t} ({zt}cos to word {float(unit(r) @ p):.2f}): "
                          f"{', '.join(emb.words[i] for i in idx[:12])}")
                    r = unit(r) - float(unit(r) @ s) * s
    if args.csv and rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("K", "alpha", "word1", "word2"), k))
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
        sp = args.csv.rsplit(".", 1)[0] + "_summary.csv"
        with open(sp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)
        print(f"wrote {sp}")
        dp = args.csv.rsplit(".", 1)[0] + "_detect.csv"
        with open(dp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detsum[0].keys()))
            w.writeheader(); w.writerows(detsum)
        print(f"wrote {dp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
