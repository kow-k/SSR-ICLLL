#!/usr/bin/env python3
"""
stage1_anchor.py

STAGE 1 (anchor, no SSR, no embeddings): for a translation pair (x_en, y_fr),
extract M = the WordNet synset they genuinely share via ILI (x has an English
lemma, y has a French lemma in the same Princeton synset). M is the GOLD sense
that x and y correspond through; Stage 2 will ask whether SSR recovers it.

This is the GATE. It answers, on WordNet alone:
  - existence : does a shared synset M exist for the pair? (yes by construction
                if y is chosen as an ILI-sharer, but we report multiplicity)
  - ambiguity : is M unique (|shared|==1) or do x,y share several synsets?
  - testability: is x polysemous (|S|>=2)? monosemous x -> nothing to separate
  - separability: how distinct is M's member set from x's OTHER synsets'
                members? If M overlaps x's other senses heavily, even a perfect
                SSR cluster can't be attributed to M -> Stage 2 is ill-posed.

Members of a synset = its English lemmas + lemmas of direct hypernyms/hyponyms
+ gloss content words (a "synset-like word set", the WordNet analogue of an
SSR neighbor cluster). Separability = 1 - max Jaccard(M, other sense of x).

  python3 stage1_anchor.py                 # full run on WordNet
  python3 stage1_anchor.py --max-pairs 500
"""

import argparse, re
import numpy as np

POS = "n"
STOP = set("""a an the of to in on for and or but with without into from by as at
be is are was were been being this that these those it its their his her your our
my one ones such very more most also which who whom whose than then so not no nor
used often usually especially etc""".split())

def content_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP and len(w) > 2}

def members(syn):
    """English 'synset-like word set': own lemmas + hyper/hyponym lemmas +
    gloss content words. Lowercased, underscores split."""
    ms = set()
    def add_lemmas(s):
        for lem in s.lemmas():
            for part in lem.name().split("_"):
                ms.add(part.lower())
    add_lemmas(syn)
    for rel in syn.hypernyms() + syn.hyponyms():
        add_lemmas(rel)
    ms |= content_words(syn.definition())
    return ms

def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def best_translation(x, wn, lang="fra"):
    """French lemma sharing the most synsets with x (ILI). Returns
    (y, shared_synsets, S) or None."""
    S = wn.synsets(x, pos=POS)
    if not S:
        return None
    Sid = {s.name(): s for s in S}
    cand = {}
    for s in S:
        for fl in s.lemma_names(lang):
            cand.setdefault(fl.lower(), 0)
    best = None
    for y in cand:
        shared = [Sid[t.name()] for t in wn.synsets(y, lang=lang, pos=POS) if t.name() in Sid]
        if not shared:
            continue
        if best is None or len(shared) > len(best[1]):
            best = (y, shared, S)
    return best

def pick_M(shared):
    """Single-best M = the most frequent (first in WordNet order) shared synset."""
    return shared[0]

def run(wn, max_pairs=None):
    rng = np.random.default_rng(0)
    n_pairs = 0
    exists = uniq = polysemous = testable = 0
    sep_scores, maxov_scores, Ssizes, sharedsizes = [], [], [], []

    for x in sorted({l for l in wn.all_lemma_names(pos=POS)}):
        if not x.isalpha() or len(x) < 3:
            continue
        S = wn.synsets(x, pos=POS)
        if len(S) < 1:
            continue
        tr = best_translation(x, wn)
        if tr is None:
            continue
        y, shared, S = tr
        n_pairs += 1
        exists += 1                       # shared is non-empty by construction
        sharedsizes.append(len(shared))
        Ssizes.append(len(S))
        if len(shared) == 1:
            uniq += 1
        if len(S) >= 2:
            polysemous += 1
            M = pick_M(shared)
            Mmem = members(M)
            others = [members(o) for o in S if o.name() != M.name()]
            others = [o for o in others if o]
            if Mmem and others:
                maxov = max(jaccard(Mmem, o) for o in others)
                maxov_scores.append(maxov)
                sep_scores.append(1 - maxov)
                # testable = polysemous, M reasonably separable from other senses
                if (1 - maxov) >= 0.5:
                    testable += 1
        if max_pairs and n_pairs >= max_pairs:
            break

    print(f"translation pairs examined: {n_pairs}\n")
    print(f"M exists (>=1 shared synset)     : {exists:5d}  ({exists/n_pairs:.0%})")
    print(f"M unambiguous (exactly 1 shared) : {uniq:5d}  ({uniq/n_pairs:.0%})")
    print(f"x polysemous (|S|>=2, testable)  : {polysemous:5d}  ({polysemous/n_pairs:.0%})")
    print(f"  of those, M separable (sep>=.5): {testable:5d}  "
          f"({testable/max(polysemous,1):.0%} of polysemous)")

    Ssz = np.array(Ssizes); shz = np.array(sharedsizes)
    print(f"\n|S| (senses of x)      : median={np.median(Ssz):.0f}  mean={Ssz.mean():.1f}  max={Ssz.max()}")
    print(f"|shared| (x,y ILI)     : median={np.median(shz):.0f}  mean={shz.mean():.2f}  "
          f">1: {np.mean(shz>1):.0%}")

    if sep_scores:
        sep = np.array(sep_scores); mo = np.array(maxov_scores)
        print(f"\nseparability of M from x's OTHER senses (1 - max Jaccard):")
        print(f"  median={np.median(sep):.3f}  mean={sep.mean():.3f}")
        for lo, hi, lab in [(0.0,0.3,'entangled '),(0.3,0.5,'weak      '),
                            (0.5,0.8,'separable '),(0.8,1.01,'clean     ')]:
            m = ((sep>=lo)&(sep<hi)).mean()
            print(f"    sep[{lo:.1f},{hi:.1f}) {lab}: {m:5.0%}")
        print(f"\n  >>> GATE: {testable} polysemous pairs have a clean, separable M.")
        print(f"      These are where Stage 2 (SSR vs M) is well-posed.")
        print(f"      Median max-overlap with a sibling sense = {np.median(mo):.3f} "
              f"(the distractor floor SSR must beat).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()
    import nltk
    for r in ("wordnet", "omw-1.4"):
        nltk.download(r, quiet=True)
    from nltk.corpus import wordnet as wn
    wn.synsets("dog")
    run(wn, args.max_pairs)

if __name__ == "__main__":
    main()
