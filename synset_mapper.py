"""
synset_mapper.py — Map separated sense vectors to WordNet synsets.

Given a polysemous word and its separated sense vectors (from SenseExplorer),
this module finds the best-matching WordNet synset for each sense by measuring
how well the sense vector's nearest neighbors are "covered" by each synset's
expanded vocabulary.

v2 improvements:
  - Asymmetric lemmatization: neighbor words are lemmatized to match WordNet
    citation forms, but WordNet's own forms are preserved as-is (plurality
    can encode semantic distinctions in the lexicon).
  - Incremental k: coverage is computed at k=5,10,15,...,max_k, producing
    a coverage curve that reveals matching robustness. A good match shows
    high coverage at small k that degrades gracefully.
  - WordNet OOV filtering: neighbors not in WordNet vocabulary are discarded
    before matching (effectively removes proper nouns and corpus artifacts).

Usage:
    from synset_mapper import SynsetMapper
    mapper = SynsetMapper(embeddings)
    results = mapper.map_senses("bank", sense_vectors)
    mapper.report(results)

Requires: nltk with wordnet corpus, numpy
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

try:
    from nltk.corpus import wordnet as wn
    wn.synsets('test')
    HAS_WORDNET = True
except (ImportError, LookupError, OSError):
    HAS_WORDNET = False
    wn = None
    print("Warning: NLTK WordNet not available.")

# Try to use NLTK's lemmatizer; fall back to a simple suffix-based one
try:
    from nltk.stem import WordNetLemmatizer
    _wnl = WordNetLemmatizer()
    HAS_LEMMATIZER = True
except (ImportError, LookupError):
    HAS_LEMMATIZER = False


def lemmatize(word: str) -> Set[str]:
    """Lemmatize a word, returning all possible lemmas across POS.

    We generate lemmas for all four POS (noun, verb, adj, adv) because
    we don't know the POS of the neighbor word in context. The union of
    all lemma forms maximizes the chance of matching a WordNet entry.

    Args:
        word: A lowercase word token from the embedding vocabulary.

    Returns:
        Set of possible lemma forms (always includes the original).
    """
    forms = {word}
    if HAS_LEMMATIZER:
        for pos in ('n', 'v', 'a', 'r'):
            forms.add(_wnl.lemmatize(word, pos))
    else:
        # Simple suffix-based fallback
        if word.endswith('ies') and len(word) > 4:
            forms.add(word[:-3] + 'y')
        if word.endswith('es') and len(word) > 3:
            forms.add(word[:-2])
        if word.endswith('s') and not word.endswith('ss') and len(word) > 2:
            forms.add(word[:-1])
        if word.endswith('ing') and len(word) > 4:
            forms.add(word[:-3])
            forms.add(word[:-3] + 'e')
        if word.endswith('ed') and len(word) > 3:
            forms.add(word[:-2])
            forms.add(word[:-1])
        if word.endswith('er') and len(word) > 3:
            forms.add(word[:-2])
            forms.add(word[:-1])
    return forms


def is_in_wordnet(word: str) -> bool:
    """Check if any lemma form of a word exists in WordNet.

    Used as a proxy for proper noun filtering: words unknown to
    WordNet are likely proper nouns, abbreviations, or corpus artifacts.
    """
    if not HAS_WORDNET:
        return True
    for form in lemmatize(word):
        if wn.synsets(form):
            return True
    return False


class SynsetMapper:
    """Maps separated sense vectors to WordNet synsets via neighbor coverage.

    v2: Asymmetric lemmatization, incremental k, WordNet OOV filtering.
    """

    def __init__(self, embeddings: Dict[str, np.ndarray],
                 max_k: int = 50, k_step: int = 5, min_k: int = 5):
        """
        Args:
            embeddings: Dictionary mapping words to their embedding vectors.
            max_k: Maximum number of nearest neighbors to retrieve.
            k_step: Step size for incremental k evaluation.
            min_k: Starting k for incremental evaluation.
        """
        self.embeddings = embeddings
        self.max_k = max_k
        self.k_step = k_step
        self.min_k = min_k

        # Pre-compute normalized embeddings for fast cosine similarity
        self.vocab = list(embeddings.keys())
        self.matrix = np.array([embeddings[w] for w in self.vocab])
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.normed_matrix = self.matrix / norms

        # Build word-to-index lookup
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}

        # Caches
        self._wn_cache: Dict[str, bool] = {}
        self._lemma_cache: Dict[str, Set[str]] = {}
        self._pos_cache: Dict[str, Dict[str, int]] = {}  # word → {pos: synset_count}

    def _is_in_wordnet(self, word: str) -> bool:
        """Cached WordNet membership check."""
        if word not in self._wn_cache:
            self._wn_cache[word] = is_in_wordnet(word)
        return self._wn_cache[word]

    def _lemmatize(self, word: str) -> Set[str]:
        """Cached lemmatization."""
        if word not in self._lemma_cache:
            self._lemma_cache[word] = lemmatize(word)
        return self._lemma_cache[word]

    def _word_pos_counts(self, word: str) -> Dict[str, int]:
        """Get the number of WordNet synsets per POS for a word.

        Tries all lemma forms of the word and aggregates counts.
        Cached for efficiency.

        Args:
            word: A lowercase word token.

        Returns:
            Dict mapping POS tag ('n','v','a','r') to synset count.
        """
        if word in self._pos_cache:
            return self._pos_cache[word]

        counts: Dict[str, int] = {}
        if HAS_WORDNET:
            for form in self._lemmatize(word):
                for ss in wn.synsets(form):
                    p = ss.pos()
                    # WordNet uses 's' for adjective satellite; merge with 'a'
                    if p == 's':
                        p = 'a'
                    counts[p] = counts.get(p, 0) + 1

        self._pos_cache[word] = counts
        return counts

    POS_LABELS = {'n': 'noun', 'v': 'verb', 'a': 'adj', 'r': 'adv'}

    def infer_pos(self, neighbors: List[Tuple[str, float]],
                  top_n: int = 15) -> Dict[str, float]:
        """Infer the POS of a sense from its nearest neighbors.

        For each neighbor, query WordNet for its synset-count-per-POS,
        weighted by cosine similarity. Aggregate into a soft distribution
        over POS tags. The dominant POS tells us whether this sense
        vector lives in nominal, verbal, or adjectival space.

        Args:
            neighbors: List of (word, cosine_similarity) tuples.
            top_n: How many top neighbors to use for inference.

        Returns:
            Dict mapping POS tag to probability (sums to 1.0).
            Includes 'dominant' key with the winning POS tag.
        """
        weighted_counts: Dict[str, float] = {}

        for word, sim in neighbors[:top_n]:
            pos_counts = self._word_pos_counts(word)
            total = sum(pos_counts.values())
            if total == 0:
                continue
            # Each neighbor votes proportionally to its POS distribution,
            # weighted by its similarity to the sense vector
            for pos, count in pos_counts.items():
                frac = count / total
                weighted_counts[pos] = weighted_counts.get(pos, 0) + sim * frac

        total_weight = sum(weighted_counts.values())
        if total_weight == 0:
            return {'n': 1.0, 'dominant': 'n', 'confidence': 0.0}

        dist = {pos: w / total_weight for pos, w in weighted_counts.items()}
        dominant = max(dist, key=dist.get)
        dist['dominant'] = dominant
        dist['confidence'] = dist[dominant]

        return dist

    def get_neighbors(self, vector: np.ndarray,
                      exclude: Optional[Set[str]] = None,
                      top_k: Optional[int] = None,
                      filter_oov: bool = True) -> List[Tuple[str, float]]:
        """Find nearest neighbors, optionally filtering WordNet OOV words.

        Args:
            vector: The query vector.
            exclude: Set of words to exclude (e.g., the target word).
            top_k: Number of neighbors to return.
            filter_oov: If True, discard words not in WordNet vocabulary.

        Returns:
            List of (word, cosine_similarity) tuples, sorted by similarity.
        """
        if top_k is None:
            top_k = self.max_k
        if exclude is None:
            exclude = set()

        norm = np.linalg.norm(vector)
        if norm == 0:
            return []
        normed_vec = vector / norm

        sims = self.normed_matrix @ normed_vec

        # Fetch extra to account for exclusions and OOV filtering.
        # Large vocabularies (e.g., Google News 3M) have many proper nouns,
        # multi-word entries, and capitalized words that fail the WordNet
        # OOV filter, so we need to fetch proportionally more candidates.
        base_fetch = top_k * 3 + len(exclude) + 50
        if filter_oov and len(self.vocab) > 500000:
            # For large vocabs, scale up to ensure enough WordNet words
            n_fetch = max(base_fetch, top_k * 10 + 200)
        else:
            n_fetch = base_fetch
        top_indices = np.argsort(sims)[::-1][:n_fetch]

        results = []
        n_filtered = 0
        for idx in top_indices:
            word = self.vocab[idx]
            if word in exclude:
                continue
            if filter_oov and not self._is_in_wordnet(word):
                n_filtered += 1
                continue
            results.append((word, float(sims[idx])))
            if len(results) >= top_k:
                break

        return results

    def expand_synset(self, synset, depth: int = 2) -> Set[str]:
        """Expand a synset into a set of related words.

        Words are kept in their original WordNet forms (NO lemmatization)
        because plurality and inflection can encode semantic distinctions
        in the lexicon.

        Collects from: lemma names, hypernyms (1 level), hyponyms (to depth),
        similar_tos, also_sees, gloss words, example words.

        Args:
            synset: A WordNet synset object.
            depth: How many levels of hyponyms to traverse.

        Returns:
            Set of lowercase words associated with this synset.
        """
        words = set()

        def add_lemma_names(ss):
            for lemma in ss.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                words.add(name)
                for part in name.split():
                    words.add(part)

        # Direct lemma names
        add_lemma_names(synset)

        # Hypernyms (1 level up)
        for hyper in synset.hypernyms():
            add_lemma_names(hyper)

        # Hyponyms (recursive to depth)
        def collect_hyponyms(ss, d):
            if d <= 0:
                return
            for hypo in ss.hyponyms():
                add_lemma_names(hypo)
                collect_hyponyms(hypo, d - 1)

        collect_hyponyms(synset, depth)

        # Similar-to and also-see relations
        for related in synset.similar_tos() + synset.also_sees():
            add_lemma_names(related)

        # Gloss words (filter stopwords and short words)
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'that', 'this', 'these', 'those',
            'or', 'and', 'but', 'if', 'while', 'because', 'until', 'so',
            'not', 'no', 'nor', 'only', 'own', 'same', 'than', 'too',
            'very', 'just', 'about', 'such', 'it', 'its', 'which', 'who',
            'whom', 'what', 'where', 'when', 'how', 'all', 'each', 'every',
            'both', 'few', 'more', 'most', 'other', 'some', 'any', 'etc',
        }
        for text_source in [synset.definition()] + synset.examples():
            for word in text_source.split():
                cleaned = word.strip('.,;:!?()[]"\'').lower()
                if len(cleaned) > 2 and cleaned not in stopwords:
                    words.add(cleaned)

        return words

    def _match_word(self, neighbor_word: str, synset_words: Set[str]) -> bool:
        """Check if a neighbor word matches any word in the synset set.

        Asymmetric lemmatization: the neighbor word is lemmatized to
        produce multiple candidate forms, which are then checked against
        the synset words (kept in original form).

        Args:
            neighbor_word: A word from the sense vector's neighbors.
            synset_words: Expanded word set for a synset (not lemmatized).

        Returns:
            True if any lemma form of the neighbor matches a synset word.
        """
        lemma_forms = self._lemmatize(neighbor_word)
        return bool(lemma_forms & synset_words)

    def compute_coverage(self, neighbors: List[Tuple[str, float]],
                         synset_words: Set[str]) -> dict:
        """Compute how well a synset covers a set of neighbors.

        Uses asymmetric lemmatization: neighbor words are lemmatized
        before matching against synset words (kept as-is).

        Args:
            neighbors: List of (word, similarity) from sense vector neighbors.
            synset_words: Expanded word set for a synset.

        Returns:
            Dictionary with coverage metrics.
        """
        matched = []
        unmatched = []

        for rank, (word, sim) in enumerate(neighbors):
            if self._match_word(word, synset_words):
                matched.append((word, sim, rank))
            else:
                unmatched.append((word, sim, rank))

        n_total = len(neighbors)
        n_matched = len(matched)

        coverage = n_matched / n_total if n_total > 0 else 0.0

        if neighbors:
            total_sim = sum(sim for _, sim in neighbors)
            matched_sim = sum(sim for _, sim, _ in matched)
            sim_coverage = matched_sim / total_sim if total_sim > 0 else 0.0
        else:
            sim_coverage = 0.0

        return {
            'coverage': coverage,
            'sim_coverage': sim_coverage,
            'n_matched': n_matched,
            'n_total': n_total,
            'matched_words': [(w, s) for w, s, _ in matched],
            'unmatched_words': [(w, s) for w, s, _ in unmatched],
        }

    def compute_coverage_curve(self, neighbors: List[Tuple[str, float]],
                               synset_words: Set[str]) -> List[dict]:
        """Compute coverage at incremental k values.

        Starting from min_k neighbors, step by k_step up to the full
        neighbor list. At each k, compute coverage using only the top-k
        neighbors. This reveals how robust the matching is: a good match
        shows high coverage at small k that degrades gracefully.

        Args:
            neighbors: Full list of (word, similarity) neighbors.
            synset_words: Expanded word set for a synset.

        Returns:
            List of dicts, one per k value, each containing coverage metrics
            plus the 'k' value.
        """
        curve = []
        k_values = list(range(self.min_k, len(neighbors) + 1, self.k_step))
        # Always include the full list
        if k_values and k_values[-1] != len(neighbors):
            k_values.append(len(neighbors))

        for k in k_values:
            top_k_neighbors = neighbors[:k]
            cov = self.compute_coverage(top_k_neighbors, synset_words)
            cov['k'] = k
            curve.append(cov)

        return curve

    def map_senses(self, word: str, sense_vectors: List[np.ndarray],
                   pos: Optional[str] = None,
                   hyponym_depth: int = 2,
                   filter_oov: bool = True,
                   infer_pos: bool = False,
                   pos_threshold: float = 0.5) -> dict:
        """Map sense vectors to WordNet synsets with incremental k evaluation.

        Args:
            word: The target polysemous word.
            sense_vectors: List of separated sense vectors from SenseExplorer.
            pos: Optional POS filter applied to ALL senses ('n', 'v', 'a', 'r').
                 Ignored if infer_pos is True.
            hyponym_depth: How deep to traverse hyponyms when expanding synsets.
            filter_oov: If True, discard neighbors not in WordNet vocabulary.
            infer_pos: If True, infer POS per sense from neighbors and filter
                       synsets to the dominant POS. Overrides the pos parameter.
            pos_threshold: Minimum confidence for POS inference to apply
                           filtering. Below this, all synsets are kept.

        Returns:
            Dictionary with mapping results including coverage curves.
        """
        if not HAS_WORDNET:
            raise RuntimeError("WordNet not available.")

        # Get all synsets for the word (unfiltered; per-sense filtering below)
        all_synsets = wn.synsets(word)
        if not all_synsets:
            raise ValueError(f"No WordNet synsets found for '{word}'")

        # If a fixed POS is given (and not inferring), pre-filter
        if pos and not infer_pos:
            all_synsets = [ss for ss in all_synsets if ss.pos() in (pos, 's' if pos == 'a' else '')]
            if not all_synsets:
                raise ValueError(f"No WordNet synsets for '{word}' with POS={pos}")

        # Pre-expand all synsets (we may use different subsets per sense)
        synset_expansions = {}
        for ss in all_synsets:
            synset_expansions[ss] = self.expand_synset(ss, depth=hyponym_depth)

        results = {
            'word': word,
            'n_senses': len(sense_vectors),
            'n_synsets_total': len(all_synsets),
            'n_synsets': len(all_synsets),  # backward compat
            'sense_mappings': [],
            'filter_oov': filter_oov,
            'pos_inference': infer_pos,
        }

        for i, svec in enumerate(sense_vectors):
            neighbors = self.get_neighbors(svec, exclude={word},
                                           top_k=self.max_k,
                                           filter_oov=filter_oov)

            # Per-sense POS inference
            pos_dist = None
            synsets_for_sense = all_synsets
            if infer_pos:
                pos_dist = self.infer_pos(neighbors)
                dominant_pos = pos_dist['dominant']
                confidence = pos_dist['confidence']

                if confidence >= pos_threshold:
                    # Filter synsets to dominant POS
                    filtered = [ss for ss in all_synsets
                                if ss.pos() == dominant_pos or
                                (dominant_pos == 'a' and ss.pos() == 's')]
                    if filtered:
                        synsets_for_sense = filtered

            sense_result = {
                'sense_idx': i,
                'neighbors': neighbors[:20],
                'n_neighbors': len(neighbors),
                'pos_distribution': pos_dist,
                'n_synsets_filtered': len(synsets_for_sense),
                'synset_scores': [],
            }

            for ss in synsets_for_sense:
                ss_words = synset_expansions[ss]

                # Coverage at full k
                full_cov = self.compute_coverage(neighbors, ss_words)

                # Coverage curve (incremental k)
                curve = self.compute_coverage_curve(neighbors, ss_words)

                # Best coverage across all k values
                if curve:
                    best_k_cov = max(curve, key=lambda c: c['coverage'])
                else:
                    best_k_cov = dict(full_cov)
                    best_k_cov['k'] = full_cov['n_total']  # full neighbor list as k

                sense_result['synset_scores'].append({
                    'synset': ss,
                    'synset_name': ss.name(),
                    'definition': ss.definition(),
                    'expanded_size': len(ss_words),
                    # Full-k metrics
                    'coverage': full_cov['coverage'],
                    'sim_coverage': full_cov['sim_coverage'],
                    'n_matched': full_cov['n_matched'],
                    'n_total': full_cov['n_total'],
                    'matched_words': full_cov['matched_words'],
                    'unmatched_words': full_cov['unmatched_words'],
                    # Coverage curve
                    'coverage_curve': curve,
                    # Best-k metrics
                    'best_k': best_k_cov['k'],
                    'best_k_coverage': best_k_cov['coverage'],
                    'best_k_matched': best_k_cov['n_matched'],
                    'best_k_total': best_k_cov['n_total'],
                })

            # Sort by best-k coverage (most informative ranking)
            sense_result['synset_scores'].sort(
                key=lambda x: x['best_k_coverage'], reverse=True)

            if sense_result['synset_scores']:
                sense_result['best_match'] = sense_result['synset_scores'][0]
            else:
                sense_result['best_match'] = None

            results['sense_mappings'].append(sense_result)

        # Check for conflicts
        best_synsets = [m['best_match']['synset_name']
                        for m in results['sense_mappings']
                        if m['best_match'] is not None]
        results['has_conflicts'] = len(best_synsets) != len(set(best_synsets))

        # If conflicts exist, attempt hyponym drill-down resolution
        if results['has_conflicts']:
            self._resolve_conflicts(results, sense_vectors, word,
                                    hyponym_depth, filter_oov)

        return results

    def _resolve_conflicts(self, results: dict,
                           sense_vectors: List[np.ndarray],
                           word: str,
                           hyponym_depth: int,
                           filter_oov: bool):
        """Resolve conflicts where multiple senses map to the same synset.

        When two or more separated senses converge on a single synset,
        that synset is too coarse — it subsumes a distinction the embedding
        space captures. We drill into the hyponyms of the shared synset
        and re-match each conflicting sense against individual hyponyms.

        This implements hierarchical refinement: match coarse first, then
        zoom into the taxonomy to find fine-grained distinctions.

        Args:
            results: The mapping results dict (modified in place).
            sense_vectors: The original sense vectors.
            word: The target word.
            hyponym_depth: Depth for expanding hyponym synsets.
            filter_oov: Whether to filter OOV neighbors.
        """
        # Find conflict groups: synset_name → [sense indices]
        synset_to_senses = defaultdict(list)
        for mapping in results['sense_mappings']:
            if mapping['best_match'] is not None:
                ss_name = mapping['best_match']['synset_name']
                synset_to_senses[ss_name].append(mapping['sense_idx'])

        conflict_groups = {ss: indices for ss, indices in synset_to_senses.items()
                          if len(indices) > 1}

        if not conflict_groups:
            return

        results['conflict_resolutions'] = []

        for shared_ss_name, sense_indices in conflict_groups.items():
            # Get the shared synset object
            shared_ss = None
            for mapping in results['sense_mappings']:
                if (mapping['best_match'] is not None and
                    mapping['best_match']['synset_name'] == shared_ss_name):
                    shared_ss = mapping['best_match']['synset']
                    break

            if shared_ss is None:
                continue

            # Get direct hyponyms of the shared synset
            hyponyms = shared_ss.hyponyms()
            if not hyponyms:
                # No hyponyms to drill into
                results['conflict_resolutions'].append({
                    'shared_synset': shared_ss_name,
                    'shared_definition': shared_ss.definition(),
                    'sense_indices': sense_indices,
                    'resolved': False,
                    'reason': 'no hyponyms available',
                })
                continue

            # Expand each hyponym individually
            hypo_expansions = {}
            for hypo in hyponyms:
                hypo_expansions[hypo] = self.expand_synset(hypo, depth=max(0, hyponym_depth - 1))

            # Re-match each conflicting sense against hyponyms
            sense_hypo_scores = {}
            for si in sense_indices:
                svec = sense_vectors[si]
                neighbors = self.get_neighbors(svec, exclude={word},
                                               top_k=self.max_k,
                                               filter_oov=filter_oov)

                hypo_scores = []
                for hypo in hyponyms:
                    hypo_words = hypo_expansions[hypo]
                    curve = self.compute_coverage_curve(neighbors, hypo_words)
                    best_k_cov = max(curve, key=lambda c: c['coverage']) if curve else {
                        'coverage': 0, 'k': 0, 'n_matched': 0, 'n_total': 0
                    }
                    full_cov = self.compute_coverage(neighbors, hypo_words)

                    hypo_scores.append({
                        'synset': hypo,
                        'synset_name': hypo.name(),
                        'definition': hypo.definition(),
                        'expanded_size': len(hypo_words),
                        'coverage': full_cov['coverage'],
                        'n_matched': full_cov['n_matched'],
                        'n_total': full_cov['n_total'],
                        'matched_words': full_cov['matched_words'],
                        'coverage_curve': curve,
                        'best_k': best_k_cov['k'],
                        'best_k_coverage': best_k_cov['coverage'],
                        'best_k_matched': best_k_cov.get('n_matched', 0),
                        'best_k_total': best_k_cov.get('n_total', 0),
                    })

                hypo_scores.sort(key=lambda x: x['best_k_coverage'], reverse=True)
                sense_hypo_scores[si] = hypo_scores

            # Check if drill-down resolves the conflict
            best_hypos = {}
            for si in sense_indices:
                scores = sense_hypo_scores[si]
                if scores and scores[0]['best_k_coverage'] > 0:
                    best_hypos[si] = scores[0]['synset_name']
                else:
                    best_hypos[si] = None

            resolved_names = [v for v in best_hypos.values() if v is not None]
            is_resolved = (len(set(resolved_names)) == len(resolved_names) and
                          len(resolved_names) == len(sense_indices))

            resolution = {
                'shared_synset': shared_ss_name,
                'shared_definition': shared_ss.definition(),
                'sense_indices': sense_indices,
                'n_hyponyms': len(hyponyms),
                'resolved': is_resolved,
                'sense_hypo_scores': sense_hypo_scores,
                'best_hypos': best_hypos,
            }
            results['conflict_resolutions'].append(resolution)

            # If resolved, update the sense mappings with refined matches
            if is_resolved:
                for si in sense_indices:
                    best = sense_hypo_scores[si][0]
                    mapping = results['sense_mappings'][si]
                    mapping['refined_match'] = best
                    mapping['refinement_source'] = shared_ss_name

    def report(self, results: dict, top_n_synsets: int = 3,
               top_n_neighbors: int = 15, show_curve: bool = True):
        """Print a formatted report of the sense-to-synset mapping."""
        word = results['word']
        oov_status = "ON" if results.get('filter_oov', False) else "OFF"
        pos_status = "ON" if results.get('pos_inference', False) else "OFF"
        print(f"\n{'='*70}")
        print(f"  Sense → Synset Mapping for '{word}'")
        print(f"  {results['n_senses']} separated senses, "
              f"{results.get('n_synsets_total', results['n_synsets'])} WordNet synsets")
        print(f"  OOV filtering: {oov_status} | "
              f"Asymmetric lemmatization: ON | "
              f"POS inference: {pos_status}")
        print(f"  Incremental k: {self.min_k} to {self.max_k} by {self.k_step}")
        print(f"{'='*70}")

        for mapping in results['sense_mappings']:
            idx = mapping['sense_idx']
            sense_name = mapping.get('sense_name', f'sense_{idx}')
            print(f"\n--- Sense {idx} ({sense_name}) ---")

            # Show POS inference if available
            pos_dist = mapping.get('pos_distribution')
            if pos_dist:
                parts = []
                for p in ('n', 'v', 'a', 'r'):
                    if p in pos_dist and isinstance(pos_dist[p], float):
                        parts.append(f"{self.POS_LABELS.get(p, p)}={pos_dist[p]:.0%}")
                dominant = pos_dist.get('dominant', '?')
                n_filtered = mapping.get('n_synsets_filtered', '?')
                print(f"  POS: {' '.join(parts)}  "
                      f"→ {self.POS_LABELS.get(dominant, dominant)} "
                      f"({n_filtered} synsets after filtering)")

            # Show neighbors with match markers and lemma info
            neighbors = mapping['neighbors'][:top_n_neighbors]
            best = mapping['best_match']
            if best:
                matched_words = {w for w, _ in best['matched_words']}
                print(f"  Neighbors (✓ = matched by best synset via lemma):")
                for w, s in neighbors:
                    lemmas = self._lemmatize(w)
                    lemma_info = ""
                    if lemmas != {w}:
                        extra = sorted(lemmas - {w})
                        lemma_info = f" →{','.join(extra)}"
                    marker = " ✓" if w in matched_words else ""
                    print(f"    {w} ({s:.3f}){lemma_info}{marker}")
            else:
                print(f"  Neighbors:")
                for w, s in neighbors:
                    print(f"    {w} ({s:.3f})")

            # Show top synset matches
            print(f"\n  Top synset matches:")
            for j, score in enumerate(mapping['synset_scores'][:top_n_synsets]):
                marker = " ◀ BEST" if j == 0 else ""
                print(f"    {j+1}. {score['synset_name']}: "
                      f"{score['definition']}")
                print(f"       full_k: cov={score['coverage']:.3f}  "
                      f"sim_wt={score['sim_coverage']:.3f}  "
                      f"matched={score['n_matched']}/{score['n_total']}")
                print(f"       best_k: cov={score['best_k_coverage']:.3f}  "
                      f"at k={score['best_k']}  "
                      f"matched={score['best_k_matched']}/{score['best_k_total']}"
                      f"{marker}")

                if show_curve and score['coverage_curve']:
                    points = []
                    for pt in score['coverage_curve']:
                        points.append(f"k={pt['k']}:{pt['coverage']:.2f}"
                                      f"({pt['n_matched']}/{pt['n_total']})")
                    print(f"       curve: {'  '.join(points)}")

        if results['has_conflicts']:
            print(f"\n⚠ Conflict: Multiple senses mapped to the same synset.")
            resolutions = results.get('conflict_resolutions', [])
            for res in resolutions:
                print(f"\n  Shared synset: {res['shared_synset']}")
                print(f"    \"{res['shared_definition']}\"")
                print(f"    Conflicting senses: {res['sense_indices']}")

                if not res['resolved'] and 'reason' in res:
                    print(f"    ✗ Cannot resolve: {res['reason']}")
                    continue

                n_hypo = res.get('n_hyponyms', 0)
                print(f"    Drilling into {n_hypo} hyponyms...")

                for si in res['sense_indices']:
                    scores = res['sense_hypo_scores'][si]
                    best_hypo = res['best_hypos'].get(si)
                    print(f"\n    Sense {si} → hyponym matches:")
                    for j, hs in enumerate(scores[:3]):
                        marker = ""
                        if j == 0 and hs['best_k_coverage'] > 0:
                            marker = " ◀ REFINED"
                        print(f"      {j+1}. {hs['synset_name']}: "
                              f"{hs['definition']}")
                        print(f"         best_k: cov={hs['best_k_coverage']:.3f}  "
                              f"at k={hs['best_k']}  "
                              f"matched={hs['best_k_matched']}/{hs['best_k_total']}"
                              f"{marker}")
                        if show_curve and hs.get('coverage_curve'):
                            points = []
                            for pt in hs['coverage_curve']:
                                points.append(f"k={pt['k']}:{pt['coverage']:.2f}"
                                              f"({pt['n_matched']}/{pt['n_total']})")
                            print(f"         curve: {'  '.join(points)}")

                if res['resolved']:
                    print(f"\n    ✓ Conflict resolved via hyponym drill-down:")
                    for si in res['sense_indices']:
                        hypo_name = res['best_hypos'][si]
                        hypo_def = ""
                        for hs in res['sense_hypo_scores'][si]:
                            if hs['synset_name'] == hypo_name:
                                hypo_def = hs['definition']
                                break
                        print(f"      Sense {si} → {hypo_name}: {hypo_def}")
                else:
                    print(f"\n    ✗ Conflict not fully resolved "
                          f"(hyponyms did not differentiate all senses)")

        print(f"\n{'='*70}\n")

    def summary_table(self, results: dict) -> str:
        """Return a concise summary table of the mapping."""
        lines = []
        word = results['word']
        lines.append(f"Sense-to-Synset Mapping: '{word}'")
        lines.append(f"{'Sense':<8} {'Best Synset':<35} "
                     f"{'Cov@full':<10} {'Cov@best_k':<12} "
                     f"{'best_k':<8} {'Matched':<10} {'Definition'}")
        lines.append("-" * 115)

        for mapping in results['sense_mappings']:
            best = mapping['best_match']
            refined = mapping.get('refined_match')
            if best:
                lines.append(
                    f"  {mapping['sense_idx']:<6} {best['synset_name']:<35} "
                    f"{best['coverage']:<10.3f} "
                    f"{best['best_k_coverage']:<12.3f} "
                    f"k={best['best_k']:<5} "
                    f"{best['n_matched']}/{best['n_total']:<8} "
                    f"{best['definition'][:40]}")
                if refined:
                    lines.append(
                        f"    └→   {refined['synset_name']:<35} "
                        f"{refined['coverage']:<10.3f} "
                        f"{refined['best_k_coverage']:<12.3f} "
                        f"k={refined['best_k']:<5} "
                        f"{refined['best_k_matched']}/{refined['best_k_total']:<8} "
                        f"{refined['definition'][:40]}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("synset_mapper v2 loaded.")
    print(f"  WordNet: {'available' if HAS_WORDNET else 'NOT available'}")
    print(f"  Lemmatizer: {'NLTK WordNetLemmatizer' if HAS_LEMMATIZER else 'suffix-based fallback'}")
    print("  Features: asymmetric lemmatization, incremental k, OOV filtering")
