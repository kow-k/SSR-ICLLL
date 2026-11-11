#!/usr/bin/env python3
"""
framenet_eval.py -- do SSR's neighbor clusters recover a word's FrameNet frames?

Inventory (FrameNet 1.7 via NLTK, nothing hand-picked):
  A target word's "senses" are the distinct frames evoked by its lexical units
  (e.g. bank.n -> Businesses, Relational_natural_features). Each frame is
  represented by the centroid of its OTHER single-word lexical units that are in
  the embedding vocabulary (the target itself is excluded).
Targets: words (default: nouns) whose lexical units evoke >= 2 frames, each frame
  with >= --min-members usable members; all frames of a target must be usable.

Method (fully unsupervised, as in Table II): the target's K nearest neighbors are
  partitioned spectrally into k = number of frames (or k chosen by --k-rule).
Scoring (as Table II): clusters <-> frames by centroid cosine (Hungarian); a frame
  is recovered iff its assigned cluster is closer to it than to any other frame.
Nulls:
  swap    (main) the SAME clusters scored against the frames of another target word with
          the same number of frames and no frame in common. This keeps every property of
          the clustering (compactness, spread) and breaks only its link to the word's
          meaning. Needed because frames are scored as separate items: distinct cluster
          centroids can hit several frames by chance, which a random partition (with
          near-identical centroids) cannot -- so the random-partition null is biased low.
  random  a random partition of the same neighbors (as Table II), for reference only.
Every (target, frame) is one item; reported: recovery, null, lift, McNemar (paired
  by item), and the share of targets with ALL frames recovered.

Circularity control (--exclude-lus): additionally removes from the neighbor pool
  every word that is a lexical unit of any of the target's frames, so a frame can
  only be recovered through words that FrameNet does not list for it.

Run from the repository root (needs sense_explorer/, io_utils.py, adaptive_k.py, NLTK
framenet_v17, downloaded automatically if missing; --fn-dir takes a local 1.5/1.7 copy):
  python framenet_eval.py --embedding glove.6B.100d.txt --topk 40,100 --csv fn_100d.csv --show 8
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from io_utils import Emb, _unit_rows
from adaptive_k import load_raw, partition, choose_all
from sense_explorer.spectral import spectral_clustering


def unit(v):
    return v / (np.linalg.norm(v) + 1e-10)


def lemma_of(lu_name):
    """'bank.n' -> ('bank', 'n'); multiword or odd names -> (None, None)."""
    if "." not in lu_name:
        return None, None
    lem, pos = lu_name.rsplit(".", 1)
    lem = lem.strip().lower()
    if not lem.isalpha():
        return None, None
    return lem, pos


def build_targets(fn, emb, pos_filter, min_members):
    frame_members = defaultdict(set)          # frame -> {lemma}
    word_frames = defaultdict(set)            # lemma -> {frame} (restricted POS)
    for lu in fn.lus():
        lem, pos = lemma_of(lu.name)
        if lem is None:
            continue
        frame_members[lu.frame.name].add(lem)
        if pos_filter == "all" or pos == pos_filter:
            word_frames[lem].add(lu.frame.name)
    targets = []
    for w, frames in word_frames.items():
        if len(frames) < 2 or w not in emb.index:
            continue
        ok = {}
        for f in frames:
            mem = [m for m in frame_members[f] if m != w and m in emb.index]
            if len(mem) >= min_members:
                ok[f] = mem
        if len(ok) == len(frames):            # every frame of the word is usable
            targets.append((emb.index[w], w, ok))
    targets.sort()
    return targets


def score(cents, frame_cents, names):
    """Per frame: recovered? (Table II rule, Hungarian on centroid cosine)."""
    C, F = len(cents), len(names)
    O = np.full((C, F), -1.0)
    for i, c in enumerate(cents):
        for j, f in enumerate(names):
            O[i, j] = float(c @ frame_cents[f])
    ri, ci = linear_sum_assignment(-O)
    amap = {int(c): int(r) for r, c in zip(ri, ci)}
    return {f: (j in amap and int(np.argmax(O[amap[j]])) == j) for j, f in enumerate(names)}


def swap_partner(targets, i, rng):
    """Another target with the same number of frames and no frame in common."""
    _, w, fr = targets[i]
    cand = [j for j, (_, w2, fr2) in enumerate(targets)
            if j != i and len(fr2) == len(fr) and not (set(fr2) & set(fr))]
    return targets[cand[rng.integers(len(cand))]] if cand else None


def run(emb, targets, K, exclude, k_rule, seed, R, B):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    items, words = [], []
    for ti, (rank, w, frames) in enumerate(targets):
        names = sorted(frames)
        frame_cents = {f: unit(np.mean([emb.M[emb.index[m]] for m in frames[f]], axis=0)) for f in names}
        banned = set().union(*frames.values()) if exclude else set()
        pool = [x for x in emb.neighbors(w, K + (len(banned) if exclude else 0) + 50)
                if x != w and x not in banned][:K]
        if len(pool) < 8:
            continue
        V = np.array([emb.M[emb.index[x]] for x in pool])
        k = len(names) if k_rule == "frames" else max(1, choose_all(V, rng, R, B)[k_rule])
        lab = partition(V, k)
        groups = [[x for x, l in zip(pool, lab) if l == i] for i in range(k)]
        groups = [g for g in groups if g]
        rnd = rng.integers(0, k, len(pool))
        rgroups = [[x for x, l in zip(pool, rnd) if l == i] for i in range(k)]
        rgroups = [g for g in rgroups if g]
        cent = lambda G: [unit(np.mean([emb.M[emb.index[x]] for x in g], axis=0)) for g in G]
        cg = cent(groups)
        s, n = score(cg, frame_cents, names), score(cent(rgroups), frame_cents, names)
        # swap null: the SAME clusters scored against another word's frames
        partner = swap_partner(targets, ti, rng)
        if partner is None:
            continue
        pnames = sorted(partner[2])
        pcents = {f: unit(np.mean([emb.M[emb.index[m]] for m in partner[2][f]], axis=0)) for f in pnames}
        sw = list(score(cg, pcents, pnames).values())
        for j, f in enumerate(names):
            items.append(dict(K=K, exclude=int(exclude), word=w, rank=rank, n_frames=len(names), k=k,
                              frame=f, n_members=len(frames[f]), ssr=s[f], null=n[f], swap=sw[j],
                              swap_word=partner[1]))
        words.append((w, names, groups, s))
    return items, words


def mcnemar(a, b):
    from scipy.stats import binomtest
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
    return n10, n01, p


def report(items, label):
    a = np.array([r["ssr"] for r in items]); b = np.array([r["null"] for r in items])
    c = np.array([r["swap"] for r in items])
    n10, n01, p = mcnemar(a, b)
    m10, m01, q = mcnemar(a, c)
    # bootstrap over words for SSR - swap
    rng = np.random.default_rng(0)
    W = sorted({r["word"] for r in items})
    dw = {w: [] for w in W}
    for r in items:
        dw[r["word"]].append((r["ssr"], r["swap"]))
    arr = [np.array(dw[w], float) for w in W]
    bs = []
    for _ in range(2000):
        pick = rng.integers(0, len(W), len(W))
        tot = np.concatenate([arr[i] for i in pick])
        bs.append(tot[:, 0].mean() - tot[:, 1].mean())
    byw = defaultdict(list)
    for r in items:
        byw[r["word"]].append((r["ssr"], r["null"]))
    allw = np.mean([all(x for x, _ in v) for v in byw.values()])
    alln = np.mean([all(y for _, y in v) for v in byw.values()])
    nf = np.mean([len(v) for v in byw.values()])
    print(f"\n[{label}]  {len(byw)} words, {len(items)} (word, frame) items, mean frames/word {nf:.2f}")
    print(f"  frame recovery   SSR {a.mean():.3f}")
    print(f"    vs SWAP null (same clusters, another word's frames)   {c.mean():.3f}   lift {a.mean()-c.mean():+.3f}  "
          f"95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]   McNemar {m10}/{m01}, p={q:.2g}")
    print(f"    vs random-partition null (favors any clustering; for reference)   {b.mean():.3f}   "
          f"lift {a.mean()-b.mean():+.3f}   McNemar {n10}/{n01}, p={p:.2g}")
    print(f"  all frames of a word recovered   SSR {allw:.3f}   null {alln:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--topk", default="40,100")
    ap.add_argument("--max-words", type=int, default=60000)
    ap.add_argument("--pos", default="n", help="target POS: n, v, a, or all")
    ap.add_argument("--min-members", type=int, default=3)
    ap.add_argument("--k-rule", default="frames",
                    help="frames (k = number of frames) or an adaptive_k rule, e.g. stability")
    ap.add_argument("--fn-dir", default=None, help="local FrameNet 1.7 directory (if NLTK cannot find it)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null-r", type=int, default=20)
    ap.add_argument("--stab-b", type=int, default=8)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if args.fn_dir:
        from nltk.corpus.reader.framenet import FramenetCorpusReader
        fn = FramenetCorpusReader(args.fn_dir, ["frRelation.xml", "frameIndex.xml", "fulltextIndex.xml",
                                                "luIndex.xml", "semTypes.xml"])
    else:
        import nltk
        from nltk.corpus import framenet as fn
        try:
            fn.frame_by_name("Businesses")
        except LookupError:
            print("FrameNet 1.7 not found; downloading via NLTK (framenet_v17) ...", file=sys.stderr)
            nltk.download("framenet_v17", quiet=True)
            from nltk.corpus import framenet as fn
    print(f"FrameNet: {len(fn.lus())} lexical units, {len(fn.frames())} frames")
    words, Rm = load_raw(args.embedding, args.max_words)
    emb = Emb(words, _unit_rows(Rm))
    targets = build_targets(fn, emb, args.pos, args.min_members)
    print(f"loaded {len(words)} words x {Rm.shape[1]}d from {args.embedding}")
    print(f"FrameNet targets (POS={args.pos}, >=2 frames, every frame >= {args.min_members} members in vocab): "
          f"{len(targets)}")
    rows = []
    for K in [int(t) for t in args.topk.split(",")]:
        for exclude in (False, True):
            items, wl = run(emb, targets, K, exclude, args.k_rule, args.seed, args.null_r, args.stab_b)
            report(items, f"K={K}, k={args.k_rule}, {'frame LUs EXCLUDED from pool' if exclude else 'all neighbors'}")
            if args.show and not exclude:
                for w, names, groups, s in wl[: args.show]:
                    print(f"    {w}: " + "  ".join(f"{f}{'+' if s[f] else '-'}" for f in names))
                    for g in groups:
                        print(f"        [{', '.join(g[:8])}]")
            rows.extend(items)
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
