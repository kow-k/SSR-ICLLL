"""
wordnet_anchors.py
==================
Automated per-SENSE anchor extraction from WordNet, built for later MANUAL
SCREENING (every candidate carries a provenance tag so you can trust or prune
by source).

Design decisions, grounded in Paper 2's anchor-source findings:
  * NOMINAL participants only. Paper 2: nouns anchor, predicates don't
    (raw-gloss content words -> 56%). We keep characteristic NOUNS.
  * Provenance-ranked. Paper 2: synonyms/extensional participants succeed,
    taxonomic hyponyms fail (31%, rare/OOV). So candidates are ranked:
      syn > holo/mero (participants) > gloss/example nouns > sister > hyper > hypo
  * POS-aware. WordNet is POS-partitioned, so for a multi-class word like
    'run' the verb senses and noun senses come out as separate groups -- this
    IS per-word-class anchoring, which is a stated SSR goal.
  * Coarsened. WordNet is far too fine (run: 57 synsets). We frequency-rank
    synsets (SemCor lemma counts) and keep the top few per POS, and also emit
    a POS-MERGED anchor set (the coarse word-class anchor).

Output: JSON (for the pipeline) + a human-readable table (for screening).
A --vocab file (one token/line, e.g. GloVe vocab) filters to in-vocabulary
anchors; without it, all candidates are shown so nothing is hidden pre-screen.

Needs: nltk with 'wordnet', 'omw-1.4', and a POS tagger
('averaged_perceptron_tagger'/'_eng'). Falls back to WordNet noun-membership
if the tagger is unavailable.
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

_LEM = WordNetLemmatizer()
try:
    from nltk import pos_tag
    pos_tag(["test"]); _HAVE_TAGGER = True
except Exception:
    _HAVE_TAGGER = False

STOP = set("""a an the of to in on at for with by from as is are was were be been
being it its this that these those and or but not no more most such very some any
one two three thing things kind sort type way part manner state act action""".split())

# provenance priority (lower = more trusted), encodes Paper 2's source ranking
PRIO = {"syn": 0, "holo": 1, "mero": 1, "gloss": 2, "example": 2,
        "sister": 3, "hyper": 4, "hypo": 5}


def _clean(lemma_str):
    """WordNet lemma -> list of single-word lowercase tokens (split multiword)."""
    return [t for t in re.split(r"[_\-\s]+", lemma_str.lower()) if t.isalpha()]


def _is_noun(tok, tag=None):
    if tag is not None:
        return tag.startswith("NN")
    return bool(wn.synsets(tok, pos=wn.NOUN))       # fallback: has a noun sense


def characteristic_nouns(text):
    """Nouns from a gloss/example -- the extensional participants, not the
    intensional predicates."""
    if not text:
        return []
    if _HAVE_TAGGER:
        toks = re.findall(r"[A-Za-z]+", text)
        tagged = pos_tag(toks)
        return [t.lower() for t, tag in tagged
                if _is_noun(t, tag) and t.isalpha() and len(t) > 2]
    return [t.lower() for t in re.findall(r"[A-Za-z]+", text)
            if len(t) > 2 and _is_noun(t.lower())]


def _same_word(cand, target):
    """Drop the target itself and its morphological variants."""
    if cand == target:
        return True
    for p in ("n", "v", "a", "r"):
        if _LEM.lemmatize(cand, p) == target or _LEM.lemmatize(target, p) == cand:
            return True
    return cand[:4] == target[:4] and abs(len(cand) - len(target)) <= 3


def synset_candidates(s, target):
    """Collect (token, provenance) candidates for one synset."""
    out = []

    def add(tokens, tag):
        for t in tokens:
            if t and t not in STOP and not _same_word(t, target):
                out.append((t, tag))

    for l in s.lemmas():                                    # synonyms
        add(_clean(l.name()), "syn")
    for rel, tag in ((s.part_holonyms() + s.member_holonyms()
                      + s.substance_holonyms(), "holo"),
                     (s.part_meronyms() + s.member_meronyms()
                      + s.substance_meronyms(), "mero")):
        for r in rel:
            for l in r.lemmas():
                add(_clean(l.name()), tag)
    add(characteristic_nouns(s.definition()), "gloss")
    for ex in s.examples():
        add(characteristic_nouns(ex), "example")
    for h in s.hypernyms():                                 # one level up
        for l in h.lemmas():
            add(_clean(l.name()), "hyper")
        for sis in h.hyponyms():                            # coordinate terms
            if sis != s:
                for l in sis.lemmas()[:2]:
                    add(_clean(l.name()), "sister")
    for h in s.hyponyms():                                  # concrete instances
        for l in h.lemmas()[:2]:
            add(_clean(l.name()), "hypo")
    return out


def rank_candidates(cands, k, vocab=None):
    """Dedupe keeping best provenance; drop OOV if vocab given; cap at k."""
    best = {}
    for tok, tag in cands:
        if vocab is not None and tok not in vocab:
            continue
        if tok not in best or PRIO[tag] < PRIO[best[tok]]:
            best[tok] = tag
    ranked = sorted(best.items(), key=lambda kv: (PRIO[kv[1]], kv[0]))
    return ranked[:k]


def synset_freq(s):
    return sum(l.count() for l in s.lemmas())


def extract(word, k=10, top_senses=4, vocab=None, pos_filter=("n", "v")):
    """Return per-POS, per-synset anchors + a POS-merged coarse anchor set."""
    result = {"word": word, "by_pos": {}}
    for pos in pos_filter:
        syns = wn.synsets(word, pos=pos)
        if not syns:
            continue
        syns = sorted(syns, key=synset_freq, reverse=True)[:top_senses]
        senses, merged = [], {}
        for s in syns:
            anchors = rank_candidates(synset_candidates(s, word), k, vocab)
            senses.append({"synset": s.name(), "definition": s.definition(),
                           "freq": synset_freq(s),
                           "anchors": [{"w": t, "src": g} for t, g in anchors]})
            for t, g in anchors:
                if t not in merged or PRIO[g] < PRIO[merged[t]]:
                    merged[t] = g
        merged_ranked = [t for t, _ in
                         sorted(merged.items(), key=lambda kv: (PRIO[kv[1]], kv[0]))][:k]
        result["by_pos"][pos] = {"senses": senses, "merged_anchor": merged_ranked}
    return result


def load_vocab(path):
    with open(path) as f:
        return {line.split()[0].lower() for line in f if line.strip()}


DEFAULT_TARGETS = [
    # existing 12
    "bank", "apple", "bass", "crane", "mouse", "spring", "plant", "star",
    "rock", "seal", "bat", "club",
    # Kow additions
    "book", "cell", "run",
    # Paper 4 homonyms (new)
    "bow", "light", "mole", "palm", "pen", "ring", "trunk",
    # fish cluster for the divergence probe
    "pike", "perch", "trout",
]

POS_NAME = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}


def print_table(res):
    print(f"\n{'='*70}\n{res['word'].upper()}")
    for pos, blk in res["by_pos"].items():
        print(f"  [{POS_NAME[pos]}]  merged word-class anchor: "
              f"{', '.join(blk['merged_anchor']) or '(none in vocab)'}")
        for sn in blk["senses"]:
            defn = sn["definition"][:60]
            print(f"    {sn['synset']:20} (f={sn['freq']:>3}) {defn}")
            print(f"        anchors: " +
                  ", ".join(f"{a['w']}[{a['src']}]" for a in sn["anchors"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=None, help="embedding vocab file to filter to")
    ap.add_argument("--k", type=int, default=10, help="max anchors per sense")
    ap.add_argument("--top_senses", type=int, default=4,
                    help="max synsets per POS (frequency-ranked)")
    ap.add_argument("--pos", default="nv", help="POS to include: subset of nva r")
    ap.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    ap.add_argument("--json", default=None, help="write full results as JSON here")
    a = ap.parse_args()

    vocab = load_vocab(a.vocab) if a.vocab else None
    pos_filter = tuple(a.pos)
    if not _HAVE_TAGGER:
        print("(note: POS tagger unavailable; using WordNet noun-membership fallback)")
    allres = {}
    for w in a.targets:
        res = extract(w, k=a.k, top_senses=a.top_senses, vocab=vocab,
                      pos_filter=pos_filter)
        allres[w] = res
        print_table(res)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(allres, f, indent=2)
        print(f"\nwrote {a.json}")
