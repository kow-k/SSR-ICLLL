"""
run_synset_mapping.py — Run sense separation + WordNet synset mapping.

This script:
1. Loads GloVe embeddings
2. Runs SenseExplorer sense separation on target polysemous words
3. Maps each separated sense vector to WordNet synsets
4. Produces a detailed report

Usage:
    python run_synset_mapping.py --glove /path/to/glove.6B.300d.txt
    python run_synset_mapping.py --glove /path/to/glove.6B.300d.txt --words bank bat crane
    python run_synset_mapping.py --glove /path/to/glove.bin --preset polysemy
    python run_synset_mapping.py --glove /path/to/glove.bin --preset homonym --oversplit
    python run_synset_mapping.py --glove /path/to/glove.bin --words bank mouse --oversplit 3

Requires: numpy, nltk (with wordnet), sense_explorer
"""

import argparse
import numpy as np
import sys
import time
from typing import Dict, List, Tuple, Optional

# Ensure NLTK wordnet is available
import nltk
try:
    from nltk.corpus import wordnet as wn
    wn.synsets('test')  # Verify data is loaded
except LookupError:
    print("Downloading WordNet data...")
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    from nltk.corpus import wordnet as wn

from synset_mapper import SynsetMapper


def load_embeddings(path: str, vocab_limit: int = None) -> dict:
    """Load word embeddings from various formats.

    Supported formats:
      - .txt: GloVe text format (word followed by space-separated floats)
      - .txt.gz / .gz: Gzipped GloVe text format
      - .bin: Gensim KeyedVectors binary or Word2Vec binary format
      - .model: Gensim native saved model (KeyedVectors.load)

    Args:
        path: Path to the embedding file.
        vocab_limit: If set, only load the first N words.

    Returns:
        Dictionary mapping words to numpy vectors.
    """
    import os
    ext = os.path.splitext(path)[1].lower()

    # Handle double extensions like .txt.gz
    if ext == '.gz':
        # Peek inside to determine if text or binary
        import gzip
        with gzip.open(path, 'rb') as f:
            sample = f.read(100)
        # Word2vec binary: header line is text ("3000000 300\n") followed
        # by binary floats which contain non-UTF-8 bytes.
        # GloVe text: every line is valid UTF-8 text.
        try:
            sample.decode('utf-8')
            # All valid UTF-8 — treat as gzipped text
            return _load_glove_txt(path, vocab_limit, gzipped=True)
        except UnicodeDecodeError:
            # Contains binary data — load via gensim
            return _load_gensim_bin(path, vocab_limit)

    if ext in ('.bin', '.model'):
        return _load_gensim_bin(path, vocab_limit)
    elif ext == '.npy':
        raise ValueError(
            f"Cannot load .npy file directly. "
            f"Use the corresponding .model file instead: "
            f"{path.replace('.vectors.npy', '')}")
    else:
        return _load_glove_txt(path, vocab_limit)


def _load_glove_txt(path: str, vocab_limit: int = None,
                    gzipped: bool = False) -> dict:
    """Load GloVe embeddings from a text file (optionally gzipped).

    Automatically detects and skips word2vec-style header lines
    (e.g., '1193514 200') that gensim-redistributed files include.
    """
    import gzip
    fmt = "gzipped GloVe text" if gzipped else "GloVe text"
    print(f"Loading {fmt} embeddings from {path}...")
    embeddings = {}
    opener = gzip.open if gzipped else open
    open_kwargs = {'mode': 'rt', 'encoding': 'utf-8'} if gzipped else \
                  {'mode': 'r', 'encoding': 'utf-8'}
    count = 0
    with opener(path, **open_kwargs) as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(' ')
            # Detect word2vec-style header: exactly 2 tokens, both integers
            if i == 0 and len(parts) == 2:
                try:
                    int(parts[0])
                    int(parts[1])
                    print(f"  Detected word2vec header: {parts[0]} words, "
                          f"{parts[1]} dimensions — skipping")
                    continue
                except ValueError:
                    pass  # Not a header, treat as normal line
            if len(parts) < 3:
                continue  # Skip malformed lines
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vec
            count += 1
            if count % 100000 == 0:
                print(f"  Loaded {count} words...")
            if vocab_limit and count >= vocab_limit:
                break

    dim = len(next(iter(embeddings.values())))
    print(f"  Done: {len(embeddings)} words, {dim} dimensions")
    return embeddings


def _load_gensim_bin(path: str, vocab_limit: int = None) -> dict:
    """Load embeddings from a Gensim KeyedVectors binary file.

    Handles both native Gensim .bin format and Word2Vec binary format.
    Requires the gensim package.
    """
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        raise ImportError(
            "gensim is required for loading .bin files. "
            "Install with: pip install gensim")

    print(f"Loading Gensim binary embeddings from {path}...")

    # Try native Gensim format first, then Word2Vec binary format
    try:
        kv = KeyedVectors.load(path)
        print(f"  Loaded as native Gensim format")
    except Exception:
        try:
            kv = KeyedVectors.load_word2vec_format(path, binary=True)
            print(f"  Loaded as Word2Vec binary format")
        except Exception:
            # Some Gensim-saved models may need mmap
            kv = KeyedVectors.load(path, mmap='r')
            print(f"  Loaded as native Gensim format (mmap)")

    # Convert to dict
    vocab = kv.index_to_key
    if vocab_limit:
        vocab = vocab[:vocab_limit]

    embeddings = {}
    for i, word in enumerate(vocab):
        embeddings[word] = kv[word].astype(np.float32)
        if (i + 1) % 100000 == 0:
            print(f"  Loaded {i+1} words...")

    dim = kv.vector_size
    print(f"  Done: {len(embeddings)} words, {dim} dimensions")
    return embeddings


def run_sense_separation(word: str, embeddings: dict,
                         n_senses: int = None,
                         se_instance=None,
                         method: str = 'discover_auto',
                         force: bool = False,
                         **kwargs) -> Tuple[List[np.ndarray], Dict]:
    """Run SenseExplorer sense separation on a word.

    Uses the SenseExplorer v0.9.0+ API. Four modes available:
      - 'discover_auto': Unsupervised with automatic k (discover_senses_auto)
      - 'discover':      Unsupervised with fixed k (discover_senses)
      - 'wordnet':       WordNet-guided separation (separate_senses_wordnet)
      - 'induce':        Weakly supervised with anchors (induce_senses)

    Args:
        word: The target polysemous word.
        embeddings: GloVe embeddings dict (used only if se_instance is None).
        n_senses: Number of senses (for 'discover' mode; ignored by others).
        se_instance: Pre-initialized SenseExplorer instance (avoids re-init).
        method: Which SenseExplorer method to use.
        force: Force rediscovery even if cached (needed when sweeping
               different k values for the same word).

    Returns:
        Tuple of (list of sense vectors, sense_dict from SenseExplorer)
    """
    from sense_explorer import SenseExplorer

    # Reuse or create SenseExplorer instance
    if se_instance is None:
        se_instance = SenseExplorer(embeddings)

    # Call the appropriate method
    if method == 'discover_auto':
        sense_dict = se_instance.discover_senses_auto(word, force=force, **kwargs)
    elif method == 'discover':
        sense_dict = se_instance.discover_senses(word, n_senses=n_senses, force=force, **kwargs)
    elif method == 'wordnet':
        sense_dict = se_instance.separate_senses_wordnet(word, force=force, **kwargs)
    elif method == 'induce':
        sense_dict = se_instance.induce_senses(word, n_senses=n_senses, force=force, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Use 'discover_auto', 'discover', 'wordnet', or 'induce'.")

    # Extract sense vectors as a list (the dict values are np.ndarray)
    sense_names = sorted(sense_dict.keys())
    sense_vectors = [sense_dict[name] for name in sense_names]

    return sense_vectors, sense_dict, se_instance


def anchor_based_separation(word: str, embeddings: dict,
                            n_senses: int = None) -> list:
    """Simple anchor-based sense separation as fallback.

    Uses WordNet synsets to identify anchor words, then computes
    sense vectors as the mean of anchors for each sense.

    This is a simplified version — SenseExplorer's approach is more
    sophisticated with its simulated self-repair.
    """
    if word not in embeddings:
        raise ValueError(f"'{word}' not in embeddings")

    synsets = wn.synsets(word)
    if not synsets:
        raise ValueError(f"No synsets for '{word}'")

    # Use WordNet to get anchor words for each synset
    sense_vectors = []
    used_synsets = []

    for ss in synsets:
        # Collect words from this synset
        anchor_words = set()
        for lemma in ss.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            for part in name.split():
                if part != word and part in embeddings:
                    anchor_words.add(part)

        for hyper in ss.hypernyms():
            for lemma in hyper.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                for part in name.split():
                    if part != word and part in embeddings:
                        anchor_words.add(part)

        for hypo in ss.hyponyms()[:10]:  # Limit to avoid explosion
            for lemma in hypo.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                for part in name.split():
                    if part != word and part in embeddings:
                        anchor_words.add(part)

        if len(anchor_words) >= 2:
            vecs = [embeddings[w] for w in anchor_words]
            sense_vec = np.mean(vecs, axis=0)
            sense_vec /= np.linalg.norm(sense_vec)
            sense_vectors.append(sense_vec)
            used_synsets.append(ss.name())

    if not sense_vectors:
        raise ValueError(f"Could not build any sense vectors for '{word}'")

    print(f"  Built {len(sense_vectors)} sense vectors from synsets: "
          f"{', '.join(used_synsets)}")

    # Optionally merge very similar sense vectors
    if n_senses and len(sense_vectors) > n_senses:
        # Simple greedy merge: keep the n_senses most distinct vectors
        from itertools import combinations
        # Compute pairwise similarities
        merged = list(range(len(sense_vectors)))
        while len(set(merged)) > n_senses:
            max_sim = -1
            merge_pair = None
            groups = {}
            for i, g in enumerate(merged):
                groups.setdefault(g, []).append(i)
            group_ids = sorted(groups.keys())
            for a, b in combinations(group_ids, 2):
                va = np.mean([sense_vectors[i] for i in groups[a]], axis=0)
                vb = np.mean([sense_vectors[i] for i in groups[b]], axis=0)
                sim = va.dot(vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
                if sim > max_sim:
                    max_sim = sim
                    merge_pair = (a, b)
            if merge_pair:
                for i in range(len(merged)):
                    if merged[i] == merge_pair[1]:
                        merged[i] = merge_pair[0]

        # Rebuild sense vectors from merged groups
        groups = {}
        for i, g in enumerate(merged):
            groups.setdefault(g, []).append(i)
        new_vectors = []
        for g in sorted(groups.keys()):
            vecs = [sense_vectors[i] for i in groups[g]]
            v = np.mean(vecs, axis=0)
            v /= np.linalg.norm(v)
            new_vectors.append(v)
        sense_vectors = new_vectors

    return sense_vectors


# ── Test word sets ──────────────────────────────────────────────────
# Group 1: Classic homonyms — etymologically unrelated senses,
#   large geometric separation expected.
HOMONYM_WORDS = [
    'bank',    # financial institution / river bank (+ West Bank)
    'bat',     # animal / sports equipment
    'crane',   # bird / machine / surname
    'spring',  # season / water source / coil / jump
    'mouse',   # animal / computer device
]

# Group 2: Regular polysemy — related senses sharing etymology,
#   smaller geometric separation, gradient boundaries.
POLYSEMY_WORDS = [
    'paper',     # material / newspaper / academic article
    'head',      # body part / leader / top of something
    'line',      # rope / queue / telephone / text / boundary
    'interest',  # curiosity / financial return / stake
    'cell',      # biological / prison / phone / political unit
    'table',     # furniture / data arrangement / to postpone
    'volume',    # book / loudness / 3D space
    'board',     # plank / committee / to embark
    'root',      # plant / mathematical / origin / to cheer
    'branch',    # tree limb / division / to diverge
]

WORD_SETS = {
    'homonym':  HOMONYM_WORDS,
    'polysemy': POLYSEMY_WORDS,
    'all':      HOMONYM_WORDS + POLYSEMY_WORDS,
}

DEFAULT_WORDS = HOMONYM_WORDS


# ── Inter-sense angle diagnostics ──────────────────────────────────

def compute_inter_sense_angles(sense_vectors: List[np.ndarray]) -> List[dict]:
    """Compute pairwise angles (degrees) between all sense vectors.

    Returns list of dicts with 'i', 'j', 'angle_deg', 'cosine_sim'.
    """
    pairs = []
    n = len(sense_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            vi = sense_vectors[i]
            vj = sense_vectors[j]
            cos_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj))
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_sim))
            pairs.append({
                'i': i, 'j': j,
                'angle_deg': angle,
                'cosine_sim': float(cos_sim),
            })
    return pairs


def get_top_neighbors(sense_vec: np.ndarray, embeddings: dict,
                      top_n: int = 10) -> List[Tuple[str, float]]:
    """Get top-n nearest neighbors for a sense vector."""
    words = list(embeddings.keys())
    vecs = np.array([embeddings[w] for w in words])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs_normed = vecs / norms

    sv = sense_vec / np.linalg.norm(sense_vec)
    sims = vecs_normed @ sv
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [(words[i], float(sims[i])) for i in top_idx]


def print_oversplit_report(word: str, auto_k: int,
                           auto_vectors: List[np.ndarray],
                           auto_results: dict,
                           split_levels: List[dict],
                           embeddings: dict):
    """Print a comparative oversplit analysis.

    split_levels: list of dicts with keys:
      'k', 'vectors', 'results', 'sense_dict'
    """
    print(f"\n{'='*70}")
    print(f"  OVERSPLIT ANALYSIS: '{word}'")
    print(f"  Auto-discovered: {auto_k} senses")
    print(f"{'='*70}")

    # Show auto inter-sense angles as baseline
    auto_angles = compute_inter_sense_angles(auto_vectors)
    if auto_angles:
        print(f"\n  Baseline inter-sense angles (k={auto_k}):")
        for p in auto_angles:
            print(f"    sense_{p['i']} ↔ sense_{p['j']}: "
                  f"{p['angle_deg']:.1f}° (cos={p['cosine_sim']:.3f})")

    # For each oversplit level
    for level in split_levels:
        k = level['k']
        vectors = level['vectors']
        results = level['results']
        angles = compute_inter_sense_angles(vectors)

        print(f"\n  {'─'*60}")
        actual_n = len(vectors)
        if actual_n == k:
            print(f"  Forced k={k} (auto+{k - auto_k})")
        else:
            print(f"  Forced k={k} (auto+{k - auto_k}), got {actual_n} senses")
        print(f"  {'─'*60}")

        # Show all inter-sense angles
        print(f"  Inter-sense angles:")
        min_angle = min(p['angle_deg'] for p in angles) if angles else 0
        for p in angles:
            marker = ""
            if p['angle_deg'] < 25:
                marker = " ◀ FRAGMENTATION? (<25°)"
            elif p['angle_deg'] < 40:
                marker = " ◀ borderline"
            elif p['angle_deg'] >= 45:
                marker = " ✓ distinct"
            print(f"    sense_{p['i']} ↔ sense_{p['j']}: "
                  f"{p['angle_deg']:.1f}° (cos={p['cosine_sim']:.3f}){marker}")

        # Identify which senses are "new" — find the closest match to
        # each auto sense and flag the remainder as novel
        # Use greedy assignment: for each auto vector, find closest forced vector
        assigned = set()
        auto_to_forced = {}
        for ai, av in enumerate(auto_vectors):
            best_fi = None
            best_cos = -2
            for fi, fv in enumerate(vectors):
                if fi in assigned:
                    continue
                c = np.dot(av, fv) / (np.linalg.norm(av) * np.linalg.norm(fv))
                if c > best_cos:
                    best_cos = c
                    best_fi = fi
            if best_fi is not None:
                auto_to_forced[ai] = (best_fi, best_cos)
                assigned.add(best_fi)

        novel_senses = [i for i in range(len(vectors)) if i not in assigned]

        print(f"\n  Sense correspondence:")
        for ai, (fi, cos) in sorted(auto_to_forced.items()):
            angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            print(f"    auto sense_{ai} → forced sense_{fi} "
                  f"(angle={angle:.1f}°, cos={cos:.3f})")

        # Show novel senses with their neighborhoods
        for ni in novel_senses:
            neighbors = get_top_neighbors(vectors[ni], embeddings, top_n=10)
            print(f"\n  ★ NEW sense_{ni}:")
            print(f"    Top neighbors: {', '.join(f'{w}({s:.3f})' for w, s in neighbors)}")

            # What synset does it map to?
            if results and ni < len(results['sense_mappings']):
                sm = results['sense_mappings'][ni]
                if sm['best_match']:
                    bm = sm['best_match']
                    ss_name = (bm['synset'].name() if hasattr(bm['synset'], 'name')
                               else str(bm['synset']))
                    print(f"    Best synset: {ss_name} "
                          f"(cov@best_k={bm['best_k_coverage']:.3f} at k={bm['best_k']})")
                    print(f"    Definition: {bm['definition']}")
                else:
                    print(f"    No synset match found.")

            # Minimum angle to existing auto senses
            min_a = None
            for ai, av in enumerate(auto_vectors):
                c = np.dot(vectors[ni], av) / (
                    np.linalg.norm(vectors[ni]) * np.linalg.norm(av))
                a = np.degrees(np.arccos(np.clip(c, -1, 1)))
                if min_a is None or a < min_a:
                    min_a = a
            if min_a is not None:
                verdict = ("FRAGMENTATION" if min_a < 25 else
                           "borderline" if min_a < 40 else
                           "GENUINE candidate")
                print(f"    Min angle to auto senses: {min_a:.1f}° → {verdict}")

    # Compact comparison table
    print(f"\n  {'─'*60}")
    print(f"  Comparison table for '{word}':")
    print(f"  {'k':<5} {'Min angle':<12} {'Max cov':<12} {'Conflicts':<12} {'New synsets'}")
    print(f"  {'─'*55}")

    def _ss_name(ss):
        return ss.name() if hasattr(ss, 'name') else str(ss)

    # Auto baseline
    auto_min = min(p['angle_deg'] for p in auto_angles) if auto_angles else float('nan')
    auto_max_cov = max(m['best_match']['best_k_coverage']
                       for m in auto_results['sense_mappings']
                       if m['best_match'])
    auto_conflict = "yes" if auto_results['has_conflicts'] else "no"
    auto_synsets = set(m['best_match']['synset']
                       for m in auto_results['sense_mappings']
                       if m['best_match'])
    auto_ss_names = sorted(_ss_name(s) for s in auto_synsets)
    print(f"  {auto_k:<5} {auto_min:<12.1f} {auto_max_cov:<12.3f} "
          f"{auto_conflict:<12} {', '.join(auto_ss_names)}")

    for level in split_levels:
        k = level['k']
        actual_k = len(level['vectors'])
        results = level['results']
        angles = compute_inter_sense_angles(level['vectors'])
        lmin = min(p['angle_deg'] for p in angles) if angles else float('nan')
        lmax_cov = max(m['best_match']['best_k_coverage']
                       for m in results['sense_mappings']
                       if m['best_match']) if results else 0
        lconflict = "yes" if results and results['has_conflicts'] else "no"
        lsynsets = set(m['best_match']['synset']
                       for m in results['sense_mappings']
                       if m['best_match']) if results else set()
        new_ss = lsynsets - auto_synsets
        new_str = ', '.join(sorted(_ss_name(s) for s in new_ss)) if new_ss else "—"
        k_label = f"{k}" if actual_k == k else f"{k}→{actual_k}"
        print(f"  {k_label:<5} {lmin:<12.1f} {lmax_cov:<12.3f} "
              f"{lconflict:<12} {new_str}")


# ── WordNet-guided sweep ───────────────────────────────────────────

def wordnet_sweep_k_values(n_synsets: int) -> List[int]:
    """Compute k values for WordNet-guided sweep.

    Given N synsets, returns sorted unique k values from
    k = round(N/i) for i in [1, 1.5, 2, 2.5, 3, 4, 5, ...], descending.
    The half-steps at i=1.5 and i=2.5 fill important gaps between
    the coarsest levels (e.g., N=18: k=12 between 18 and 9).
    Filters out k < 2.
    """
    max_i = max(1, round(n_synsets / 2))
    # Build divisor sequence: 1, 1.5, 2, 2.5, 3, 4, 5, ...
    divisors = [1, 1.5, 2, 2.5] + list(range(3, max_i + 1))
    divisors = [d for d in divisors if d <= max_i]
    k_values = set()
    for d in divisors:
        k = round(n_synsets / d)
        if k >= 2:
            k_values.add(k)
    return sorted(k_values, reverse=True)


def print_wordnet_sweep_report(word: str, n_synsets: int,
                               sweep_results: List[dict],
                               embeddings: dict):
    """Print a comparative WordNet sweep analysis.

    sweep_results: list of dicts with keys:
      'requested_k', 'actual_k', 'vectors', 'results', 'angles',
      'synsets_found', 'distinct_synsets'
    """
    print(f"\n{'='*70}")
    print(f"  WORDNET SWEEP: '{word}'")
    print(f"  WordNet synsets: {n_synsets}")
    print(f"  k values tested: {', '.join(str(r['requested_k']) for r in sweep_results)}")
    print(f"{'='*70}")

    # Summary table
    print(f"\n  {'k':<8} {'actual':<8} {'Min ∠':<10} {'Mean ∠':<10} "
          f"{'Max cov':<10} {'Conflicts':<10} {'Distinct synsets'}")
    print(f"  {'─'*75}")

    for r in sweep_results:
        req_k = r['requested_k']
        act_k = r['actual_k']
        angles = r['angles']

        k_str = f"{req_k}" if act_k == req_k else f"{req_k}→{act_k}"

        # Skip levels that produced no senses
        if act_k == 0:
            print(f"  {k_str:<8} {act_k:<8} {'—':<10} {'—':<10} "
                  f"{'—':<10} {'—':<10} 0")
            continue

        if angles:
            min_a = min(p['angle_deg'] for p in angles)
            mean_a = sum(p['angle_deg'] for p in angles) / len(angles)
        else:
            min_a = float('nan')
            mean_a = float('nan')

        results = r['results']
        if results and results.get('sense_mappings'):
            coverages = [m['best_match']['best_k_coverage']
                         for m in results['sense_mappings']
                         if m['best_match']]
            max_cov = max(coverages) if coverages else 0
            conflict = "yes" if results['has_conflicts'] else "no"
            synsets = set()
            for m in results['sense_mappings']:
                if m['best_match']:
                    ss = m['best_match']['synset']
                    synsets.add(ss.name() if hasattr(ss, 'name') else str(ss))
            n_distinct = len(synsets)
        else:
            max_cov = 0
            conflict = "—"
            n_distinct = 0

        angle_marker = ""
        if angles and min_a < 25:
            angle_marker = " ⚠"
        elif angles and min_a < 40:
            angle_marker = " ~"

        min_a_str = f"{min_a:<10.1f}" if angles else f"{'—':<10}"
        mean_a_str = f"{mean_a:<10.1f}" if angles else f"{'—':<10}"
        print(f"  {k_str:<8} {act_k:<8} {min_a_str} {mean_a_str} "
              f"{max_cov:<10.3f} {conflict:<10} {n_distinct}{angle_marker}")

    # Find the "sweet spot" — highest distinct synsets with min angle ≥ 25°
    viable = [r for r in sweep_results
              if r['actual_k'] > 0 and r['angles'] and
              r['results'] and r['results'].get('sense_mappings') and
              min(p['angle_deg'] for p in r['angles']) >= 25]
    if viable:
        best = max(viable, key=lambda r: (
            len(set(m['best_match']['synset']
                    for m in r['results']['sense_mappings']
                    if m['best_match'])) if r['results'] else 0,
            min(p['angle_deg'] for p in r['angles']) if r['angles'] else 0
        ))
        best_k = best['actual_k']
        best_min = min(p['angle_deg'] for p in best['angles'])
        n_ss = len(set(m['best_match']['synset']
                       for m in best['results']['sense_mappings']
                       if m['best_match'])) if best['results'] else 0
        print(f"\n  ★ Sweet spot: k={best_k} "
              f"({n_ss} distinct synsets, min angle {best_min:.1f}°)")

    # Detail: list synsets found at each level
    print(f"\n  Synsets recovered at each k:")
    for r in sweep_results:
        req_k = r['requested_k']
        act_k = r['actual_k']
        results = r['results']
        if not results or not results.get('sense_mappings'):
            continue
        synsets = []
        for i, m in enumerate(results['sense_mappings']):
            if m['best_match']:
                ss = m['best_match']['synset']
                ss_name = ss.name() if hasattr(ss, 'name') else str(ss)
                cov = m['best_match']['best_k_coverage']
                synsets.append(f"{ss_name}({cov:.2f})")
        k_str = f"k={req_k}" if act_k == req_k else f"k={req_k}→{act_k}"
        print(f"    {k_str}: {', '.join(synsets)}")


def main():
    parser = argparse.ArgumentParser(
        description="Map separated sense vectors to WordNet synsets")
    parser.add_argument('--embeddings', '--glove', required=True,
                        dest='embeddings',
                        help='Path to embeddings file (.txt for GloVe text, '
                             '.txt.gz for gzipped GloVe, '
                             '.bin for Gensim/Word2Vec binary, '
                             '.model for Gensim native)')
    parser.add_argument('--words', nargs='+', default=None,
                        help='Words to analyze (overrides --preset)')
    parser.add_argument('--preset', choices=['homonym', 'polysemy', 'all'],
                        default=None,
                        help='Use a predefined word set: '
                             'homonym (bank,bat,...), '
                             'polysemy (paper,head,...), '
                             'all (both sets)')
    parser.add_argument('--max-k', type=int, default=50,
                        help='Maximum number of neighbors per sense (default: 50)')
    parser.add_argument('--min-k', type=int, default=5,
                        help='Starting k for incremental evaluation (default: 5)')
    parser.add_argument('--k-step', type=int, default=5,
                        help='Step size for incremental k (default: 5)')
    parser.add_argument('--hyponym-depth', type=int, default=2,
                        help='Depth of hyponym expansion (default: 2)')
    parser.add_argument('--vocab-limit', type=int, default=None,
                        help='Limit vocabulary size')
    parser.add_argument('--method',
                        choices=['discover_auto', 'discover', 'wordnet', 'induce'],
                        default='discover_auto',
                        help='Sense separation method: discover_auto (unsupervised, '
                             'default), discover (fixed k), wordnet (WordNet-guided), '
                             'or induce (anchor-guided)')
    parser.add_argument('--n-senses', type=int, default=None,
                        help='Number of senses (for discover mode; ignored by discover_auto)')
    parser.add_argument('--no-filter-oov', action='store_true',
                        help='Disable WordNet OOV filtering (keeps proper nouns etc.)')
    parser.add_argument('--infer-pos', action='store_true',
                        help='Infer POS per sense from neighbors and filter synsets')
    parser.add_argument('--oversplit', type=int, nargs='?', const=2, default=0,
                        metavar='N',
                        help='Force N extra senses beyond auto-discovered count '
                             '(default: +2 when flag is set). Reports inter-sense '
                             'angles to diagnose genuine vs. noise splits.')
    parser.add_argument('--wordnet-sweep', action='store_true',
                        help='Sweep k = round(N/i) for i=1..round(N/2) where N is '
                             'WordNet synset count. Top-down exploration from '
                             'lexicographic granularity to coarsest separation.')
    args = parser.parse_args()

    # Resolve word list: --words overrides --preset, default is homonym set
    if args.words:
        words = args.words
    elif args.preset:
        words = WORD_SETS[args.preset]
    else:
        words = HOMONYM_WORDS

    # Load embeddings
    embeddings = load_embeddings(args.embeddings, args.vocab_limit)
    mapper = SynsetMapper(embeddings, max_k=args.max_k,
                          min_k=args.min_k, k_step=args.k_step)

    # Initialize SenseExplorer once (reused across words)
    se_instance = None

    all_results = []

    # Determine which group(s) we're running
    preset_name = args.preset or ('custom' if args.words else 'homonym')
    print(f"\nWord set: {preset_name} ({len(words)} words)")

    for word in words:
        print(f"\n{'#'*70}")
        print(f"  Processing: {word}")
        print(f"{'#'*70}")

        if word not in embeddings:
            print(f"  ⚠ '{word}' not in embeddings, skipping.")
            continue

        # Check WordNet synsets
        synsets = wn.synsets(word)
        print(f"  WordNet synsets ({len(synsets)}):")
        for ss in synsets:
            print(f"    {ss.name()}: {ss.definition()}")

        # Run sense separation
        try:
            t0 = time.time()
            sense_vectors, sense_dict, se_instance = run_sense_separation(
                word, embeddings,
                n_senses=args.n_senses,
                se_instance=se_instance,
                method=args.method,
            )
            elapsed = time.time() - t0
            sense_names = sorted(sense_dict.keys())
            print(f"  Separated {len(sense_vectors)} senses "
                  f"({', '.join(sense_names)}) in {elapsed:.2f}s")
        except Exception as e:
            print(f"  ⚠ Sense separation failed: {e}")
            import traceback; traceback.print_exc()
            continue

        # Map to synsets
        try:
            results = mapper.map_senses(word, sense_vectors,
                                        hyponym_depth=args.hyponym_depth,
                                        filter_oov=not args.no_filter_oov,
                                        infer_pos=args.infer_pos)
            # Attach sense names from SenseExplorer for clearer reporting
            for i, mapping in enumerate(results['sense_mappings']):
                if i < len(sense_names):
                    mapping['sense_name'] = sense_names[i]
            mapper.report(results)
            print(mapper.summary_table(results))
            all_results.append(results)
        except Exception as e:
            print(f"  ⚠ Synset mapping failed: {e}")
            import traceback; traceback.print_exc()
            continue

        # ── Oversplit analysis ──────────────────────────────────────
        if args.oversplit > 0:
            auto_k = len(sense_vectors)
            split_levels = []

            for extra in range(1, args.oversplit + 1):
                forced_k = auto_k + extra
                print(f"\n  ── Forcing k={forced_k} (auto+{extra}) ──")

                try:
                    t0 = time.time()
                    forced_vectors, forced_dict, se_instance = run_sense_separation(
                        word, embeddings,
                        n_senses=forced_k,
                        se_instance=se_instance,
                        method='discover',
                        force=True,
                    )
                    elapsed = time.time() - t0
                    forced_names = sorted(forced_dict.keys())
                    print(f"  Separated {len(forced_vectors)} senses "
                          f"({', '.join(forced_names)}) in {elapsed:.2f}s")

                    # Map forced senses to synsets
                    forced_results = mapper.map_senses(
                        word, forced_vectors,
                        hyponym_depth=args.hyponym_depth,
                        filter_oov=not args.no_filter_oov,
                        infer_pos=args.infer_pos)
                    for i, mapping in enumerate(forced_results['sense_mappings']):
                        if i < len(forced_names):
                            mapping['sense_name'] = forced_names[i]
                    mapper.report(forced_results)
                    print(mapper.summary_table(forced_results))

                    split_levels.append({
                        'k': forced_k,
                        'vectors': forced_vectors,
                        'results': forced_results,
                        'sense_dict': forced_dict,
                    })
                except Exception as e:
                    print(f"  ⚠ Oversplit k={forced_k} failed: {e}")
                    import traceback; traceback.print_exc()

            # Print comparative oversplit report
            if split_levels:
                print_oversplit_report(
                    word, auto_k, sense_vectors, results,
                    split_levels, embeddings)

        # ── WordNet-guided sweep ────────────────────────────────────
        if args.wordnet_sweep:
            n_synsets = len(synsets)
            k_values = wordnet_sweep_k_values(n_synsets)
            print(f"\n  ── WordNet sweep: N={n_synsets} synsets → "
                  f"k values: {k_values} ──")

            sweep_results = []
            for target_k in k_values:
                divisor = round(n_synsets / target_k) if target_k > 0 else '?'
                print(f"\n  ── Sweep k={target_k} (≈N/{divisor}) ──")
                try:
                    t0 = time.time()
                    sv, sd, se_instance = run_sense_separation(
                        word, embeddings,
                        n_senses=target_k,
                        se_instance=se_instance,
                        method='discover',
                        force=True,
                    )
                    elapsed = time.time() - t0
                    sn = sorted(sd.keys())
                    actual_k = len(sv)
                    print(f"  Got {actual_k} senses "
                          f"({', '.join(sn)}) in {elapsed:.2f}s")

                    # Map to synsets
                    sr = mapper.map_senses(
                        word, sv,
                        hyponym_depth=args.hyponym_depth,
                        filter_oov=not args.no_filter_oov,
                        infer_pos=args.infer_pos)
                    for i, mapping in enumerate(sr['sense_mappings']):
                        if i < len(sn):
                            mapping['sense_name'] = sn[i]
                    mapper.report(sr)
                    print(mapper.summary_table(sr))

                    angles = compute_inter_sense_angles(sv)

                    sweep_results.append({
                        'requested_k': target_k,
                        'actual_k': actual_k,
                        'vectors': sv,
                        'results': sr,
                        'angles': angles,
                    })
                except Exception as e:
                    print(f"  ⚠ Sweep k={target_k} failed: {e}")
                    import traceback; traceback.print_exc()

            # Print sweep summary
            if sweep_results:
                print_wordnet_sweep_report(
                    word, n_synsets, sweep_results, embeddings)

    # Summary across all words
    if all_results:
        print(f"\n\n{'='*70}")
        print(f"  OVERALL SUMMARY ({preset_name})")
        print(f"{'='*70}")
        print(f"  {'Word':<12} {'Senses':<8} {'Synsets':<8} {'Conflicts':<12} "
              f"{'Best Coverage'}")
        print(f"  {'-'*65}")

        # Group labels for 'all' preset
        homonym_set = set(HOMONYM_WORDS)
        polysemy_set = set(POLYSEMY_WORDS)
        shown_homonym_header = False
        shown_polysemy_header = False

        for r in all_results:
            # Show group headers when running both sets
            if preset_name == 'all':
                w = r['word']
                if w in homonym_set and not shown_homonym_header:
                    print(f"  {'── Homonyms ──':}")
                    shown_homonym_header = True
                elif w in polysemy_set and not shown_polysemy_header:
                    print(f"  {'── Regular polysemy ──':}")
                    shown_polysemy_header = True

            best_cov = max(m['best_match']['best_k_coverage']
                          for m in r['sense_mappings']
                          if m['best_match'])
            # Determine conflict/resolution status
            if not r['has_conflicts']:
                conflict_str = "no"
            else:
                resolutions = r.get('conflict_resolutions', [])
                all_resolved = all(res['resolved'] for res in resolutions)
                if all_resolved and resolutions:
                    conflict_str = "RESOLVED"
                elif any(res['resolved'] for res in resolutions):
                    conflict_str = "PARTIAL"
                else:
                    conflict_str = "YES"
            print(f"  {r['word']:<12} {r['n_senses']:<8} {r['n_synsets']:<8} "
                  f"{conflict_str:<12} "
                  f"{best_cov:.3f}")


if __name__ == "__main__":
    main()
