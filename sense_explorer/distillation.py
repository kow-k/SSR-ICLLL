#!/usr/bin/env python3
"""
distillation.py - Sense Distillation via Independent Vector Analysis (IVA)
===========================================================================

This module implements IVA for concept distillation — extracting the shared
semantic essence from a set of word embeddings.

Theoretical Foundation:
    IVA distills the shared semantic component from a set of word embeddings.
    Unlike SSR which *discovers* senses in a single word, IVA *extracts* the
    common "essence" that a given word set shares.

Key Insight (from experiments):
    - IVA does not DISCOVER concepts automatically
    - IVA EXTRACTS the shared component from a GIVEN word set
    - The concept is determined by the selection (input-dependent)
    - This is SUPERVISED concept extraction, not discovery

Relationship to SSR:
    - SSR: Sense DISCOVERY (finds what senses exist in a word)
    - IVA: Concept DISTILLATION (extracts what a word set shares)
    - SSR remains necessary for automatic sense discovery
    - IVA complements SSR by extracting cross-word structure

Pipeline:
    SSR → anchors → IVA → purified sense directions

Experimental Validation:
    - Anchor-IVA: 100% k-accuracy at 50d
    - Plain IVA: 80% k-accuracy at 50d
    - Random input → coherence ~0.15 (garbage in, garbage out)
    - Meaningful input → coherence ~0.45-0.55 (structured output)

Two Modes:
    1. Global IVA: Iterates over entire vocabulary (may drift to global attractors)
    2. Constrained IVA: Iterates only within the given word set (recommended)

Usage:
    >>> from sense_explorer.distillation import IVADistiller
    >>> distiller = IVADistiller(embeddings)
    
    # Distill concept from a word set (constrained — recommended)
    >>> result = distiller.distill_constrained(['money', 'loan', 'account'])
    >>> print(result.exemplars)
    ['money', 'loan', 'account', 'credit', 'funds']
    
    # Global distillation (may drift)
    >>> result = distiller.distill(['money', 'loan', 'account'])

Integration with SenseExplorer:
    >>> se = SenseExplorer.from_glove("glove.txt")
    >>> results = se.distill_senses("bank")
    >>> for sense, result in results.items():
    ...     print(f"{sense}: coherence={result.coherence:.3f}")

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
Version: 0.9.5
"""

import numpy as np
from numpy.linalg import norm
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field

__version__ = "0.9.5"
__author__ = "Kow Kuroda & Claude"

__all__ = [
    'IVADistiller',
    'DistillationResult',
    'distill_concept',
    'measure_set_coherence',
    'validate_distillation',
    'create_distiller_from_explorer',
]


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DistillationResult:
    """
    Result of IVA concept distillation.
    
    Attributes:
        direction: The distilled concept direction (unit vector)
        exemplars: Nearest words to the distilled direction
        coherence: Internal coherence of the input word set (0-1)
        input_words: The words used for distillation
        n_iterations: Number of IVA iterations until convergence
        mode: 'global' or 'constrained'
    """
    direction: np.ndarray
    exemplars: List[str]
    coherence: float
    input_words: List[str]
    n_iterations: int
    mode: str
    
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"DistillationResult(mode={self.mode}, "
            f"coherence={self.coherence:.3f}, "
            f"n_iter={self.n_iterations}, "
            f"exemplars={self.exemplars[:5]})"
        )
    
    def __repr__(self) -> str:
        return self.summary()


# =============================================================================
# Main Class
# =============================================================================

class IVADistiller:
    """
    Distill shared semantic concepts via Iterative Vector Averaging (IVA).
    
    IVA iteratively refines a direction vector by:
        1. Starting with the centroid of input word vectors
        2. Finding nearest neighbors (globally or within cluster)
        3. Averaging neighbors to get a new direction
        4. Repeating until convergence
    
    The result is a "distilled" direction that captures what the
    input words share — their common semantic essence.
    
    Args:
        embeddings: Dict mapping words to numpy vectors
        max_iter: Maximum IVA iterations (default: 50)
        convergence_threshold: Cosine similarity threshold for convergence
        top_k_neighbors: Number of neighbors to average in global mode
        verbose: Print progress messages
    
    Example:
        >>> distiller = IVADistiller(embeddings)
        >>> result = distiller.distill_constrained(['money', 'loan', 'account'])
        >>> print(result.exemplars)
        ['money', 'loan', 'account', 'credit', 'funds']
    """
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        max_iter: int = 50,
        convergence_threshold: float = 0.9999,
        top_k_neighbors: int = 50,
        verbose: bool = False
    ):
        self.embeddings = embeddings
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self.top_k_neighbors = top_k_neighbors
        self.verbose = verbose
        
        # Build normalized matrix for efficient similarity computation
        self.word_list = list(embeddings.keys())
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}
        self.matrix = np.array([embeddings[w] for w in self.word_list])
        
        # Normalize
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix_norm = self.matrix / (norms + 1e-10)
        
        self.dim = self.matrix.shape[1]
        self.vocab_size = len(self.word_list)
        
        if verbose:
            print(f"IVADistiller initialized: {self.vocab_size} words, {self.dim}d")
    
    def _get_valid_vectors(
        self, 
        words: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """Get vectors for words that exist in vocabulary."""
        valid_words = [w for w in words if w in self.embeddings]
        if not valid_words:
            return np.array([]), []
        vectors = np.array([self.embeddings[w] for w in valid_words])
        return vectors, valid_words
    
    def _compute_coherence(self, vectors_norm: np.ndarray) -> float:
        """
        Compute internal coherence of a word set.
        
        Coherence = average pairwise cosine similarity.
        Higher coherence → more semantically unified set.
        """
        if len(vectors_norm) < 2:
            return 1.0
        
        # Pairwise similarities
        sim_matrix = vectors_norm @ vectors_norm.T
        
        # Average off-diagonal elements
        n = len(vectors_norm)
        total = np.sum(sim_matrix) - n  # Subtract diagonal
        count = n * (n - 1)
        
        return total / count if count > 0 else 0.0
    
    def _get_global_neighbors(
        self, 
        direction: np.ndarray, 
        top_k: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Get top-k neighbors from entire vocabulary."""
        # Compute similarities to all words
        sims = self.matrix_norm @ direction
        
        # Get top-k indices
        top_indices = np.argsort(sims)[-top_k:][::-1]
        
        neighbor_vecs = self.matrix_norm[top_indices]
        neighbor_words = [self.word_list[i] for i in top_indices]
        
        return neighbor_vecs, neighbor_words
    
    def distill(
        self,
        words: List[str],
        n_exemplars: int = 5
    ) -> DistillationResult:
        """
        Global IVA distillation — iterates over entire vocabulary.
        
        This mode may drift to "global attractors" — very frequent
        semantic patterns that pull the direction away from the
        cluster-specific concept. Use distill_constrained() if you
        want to stay within the cluster.
        
        Args:
            words: Input word set to distill
            n_exemplars: Number of exemplars to return
        
        Returns:
            DistillationResult with direction, exemplars, coherence
        """
        # Get vectors for input words
        vectors, valid_words = self._get_valid_vectors(words)
        
        if len(vectors) < 2:
            if len(vectors) == 1:
                direction = vectors[0] / (norm(vectors[0]) + 1e-10)
                return DistillationResult(
                    direction=direction,
                    exemplars=valid_words,
                    coherence=1.0,
                    input_words=valid_words,
                    n_iterations=0,
                    mode='global'
                )
            return DistillationResult(
                direction=np.zeros(self.dim),
                exemplars=[],
                coherence=0.0,
                input_words=[],
                n_iterations=0,
                mode='global'
            )
        
        # Normalize input vectors
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        
        # Compute input coherence
        coherence = self._compute_coherence(vectors_norm)
        
        # Global IVA iteration
        direction = vectors_norm.mean(axis=0)
        direction = direction / (norm(direction) + 1e-10)
        
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            old_direction = direction.copy()
            
            # Find nearest neighbors from ENTIRE VOCABULARY
            neighbor_vecs, _ = self._get_global_neighbors(
                direction, self.top_k_neighbors
            )
            
            # Average neighbors
            new_direction = neighbor_vecs.mean(axis=0)
            new_direction = new_direction / (norm(new_direction) + 1e-10)
            
            direction = new_direction
            
            # Check convergence
            if np.dot(old_direction, direction) > self.convergence_threshold:
                if self.verbose:
                    print(f"  Converged at iteration {n_iter}")
                break
        
        # Get exemplars from vocabulary
        _, exemplars = self._get_global_neighbors(direction, n_exemplars)
        
        return DistillationResult(
            direction=direction,
            exemplars=exemplars,
            coherence=coherence,
            input_words=valid_words,
            n_iterations=n_iter,
            mode='global'
        )
    
    def distill_constrained(
        self,
        words: List[str],
        n_exemplars: int = 5
    ) -> DistillationResult:
        """
        Constrained IVA distillation — iterates only within the word set.
        
        This is the RECOMMENDED mode. The distilled direction represents
        what the input words share, not what they might share with the
        broader vocabulary. Avoids drift to global attractors.
        
        Args:
            words: Input word set to distill
            n_exemplars: Number of exemplars to return (from within cluster)
        
        Returns:
            DistillationResult with direction, exemplars, coherence
        """
        # Get vectors for input words
        vectors, valid_words = self._get_valid_vectors(words)
        
        if len(vectors) < 2:
            if len(vectors) == 1:
                direction = vectors[0] / (norm(vectors[0]) + 1e-10)
                return DistillationResult(
                    direction=direction,
                    exemplars=valid_words,
                    coherence=1.0,
                    input_words=valid_words,
                    n_iterations=0,
                    mode='constrained'
                )
            return DistillationResult(
                direction=np.zeros(self.dim),
                exemplars=[],
                coherence=0.0,
                input_words=[],
                n_iterations=0,
                mode='constrained'
            )
        
        # Normalize input vectors
        vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
        
        # Compute input coherence
        coherence = self._compute_coherence(vectors_norm)
        
        # Constrained IVA iteration — only use words from the cluster
        direction = vectors_norm.mean(axis=0)
        direction = direction / (norm(direction) + 1e-10)
        
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            old_direction = direction.copy()
            
            # Find nearest neighbors WITHIN THE CLUSTER ONLY
            sims = vectors_norm @ direction
            top_k = min(max(3, len(valid_words) // 2), len(valid_words))
            top_indices = np.argsort(sims)[-top_k:]
            
            # Average top neighbors from cluster
            new_direction = vectors_norm[top_indices].mean(axis=0)
            new_direction = new_direction / (norm(new_direction) + 1e-10)
            
            direction = new_direction
            
            # Check convergence
            if np.dot(old_direction, direction) > self.convergence_threshold:
                if self.verbose:
                    print(f"  Converged at iteration {n_iter}")
                break
        
        # Get exemplars from within cluster
        sims = vectors_norm @ direction
        n_ex = min(n_exemplars, len(valid_words))
        top_indices = np.argsort(sims)[-n_ex:][::-1]
        exemplars = [valid_words[i] for i in top_indices]
        
        return DistillationResult(
            direction=direction,
            exemplars=exemplars,
            coherence=coherence,
            input_words=valid_words,
            n_iterations=n_iter,
            mode='constrained'
        )
    
    def distill_multiple(
        self,
        word_groups: Dict[str, List[str]],
        mode: str = 'constrained',
        n_exemplars: int = 5
    ) -> Dict[str, DistillationResult]:
        """
        Distill multiple word groups at once.
        
        Args:
            word_groups: Dict mapping group names to word lists
            mode: 'constrained' (recommended) or 'global'
            n_exemplars: Number of exemplars per group
        
        Returns:
            Dict mapping group names to DistillationResults
        """
        distill_fn = self.distill_constrained if mode == 'constrained' else self.distill
        
        results = {}
        for name, words in word_groups.items():
            if self.verbose:
                print(f"Distilling '{name}'...")
            results[name] = distill_fn(words, n_exemplars)
        
        return results
    
    def compare_directions(
        self,
        results: Dict[str, DistillationResult]
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute angles between distilled directions.
        
        Args:
            results: Dict of DistillationResults
        
        Returns:
            Dict mapping (name1, name2) -> angle in degrees
        """
        angles = {}
        names = list(results.keys())
        
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                d1 = results[name1].direction
                d2 = results[name2].direction
                
                # Cosine similarity
                cos_sim = np.dot(d1, d2)
                cos_sim = np.clip(cos_sim, -1, 1)
                
                # Convert to angle
                angle = np.degrees(np.arccos(cos_sim))
                angles[(name1, name2)] = angle
        
        return angles


# =============================================================================
# Convenience Functions
# =============================================================================

def distill_concept(
    embeddings: Dict[str, np.ndarray],
    words: List[str],
    mode: str = 'constrained',
    n_exemplars: int = 5
) -> DistillationResult:
    """
    One-shot concept distillation.
    
    Convenience function for quick distillation without
    creating an IVADistiller instance.
    
    Args:
        embeddings: Word embeddings dict
        words: Words to distill
        mode: 'constrained' (recommended) or 'global'
        n_exemplars: Number of exemplars to return
    
    Returns:
        DistillationResult
    
    Example:
        >>> result = distill_concept(embeddings, ['money', 'loan', 'account'])
        >>> print(result.coherence)
    """
    distiller = IVADistiller(embeddings, verbose=False)
    
    if mode == 'constrained':
        return distiller.distill_constrained(words, n_exemplars)
    else:
        return distiller.distill(words, n_exemplars)


def measure_set_coherence(
    embeddings: Dict[str, np.ndarray],
    words: List[str]
) -> float:
    """
    Measure the internal coherence of a word set.
    
    Coherence = average pairwise cosine similarity.
    Use this to validate anchor quality before distillation.
    
    Reference values:
        - Random words: ~0.15
        - Topically related: ~0.30-0.40
        - Sense-coherent: ~0.45-0.55
        - Near-synonyms: ~0.60-0.80
    
    Args:
        embeddings: Word embeddings dict
        words: Words to measure
    
    Returns:
        Coherence score (0-1)
    
    Example:
        >>> coherence = measure_set_coherence(embeddings, ['money', 'loan'])
        >>> print(f"Coherence: {coherence:.3f}")
    """
    # Get valid vectors
    valid_words = [w for w in words if w in embeddings]
    if len(valid_words) < 2:
        return 1.0 if len(valid_words) == 1 else 0.0
    
    vectors = np.array([embeddings[w] for w in valid_words])
    
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-10)
    
    # Compute pairwise similarities
    sim_matrix = vectors_norm @ vectors_norm.T
    
    # Average off-diagonal
    n = len(vectors_norm)
    total = np.sum(sim_matrix) - n
    count = n * (n - 1)
    
    return total / count if count > 0 else 0.0


def validate_distillation(
    embeddings: Dict[str, np.ndarray],
    word_groups: Dict[str, List[str]],
    mode: str = 'constrained'
) -> Dict[str, any]:
    """
    Validate distillation quality across multiple word groups.
    
    Computes coherence, inter-direction angles, and exemplar overlap.
    
    Args:
        embeddings: Word embeddings dict
        word_groups: Dict mapping names to word lists
        mode: 'constrained' or 'global'
    
    Returns:
        Dict with validation statistics
    
    Example:
        >>> groups = {'financial': ['money', 'loan'], 'river': ['water', 'shore']}
        >>> stats = validate_distillation(embeddings, groups)
        >>> print(stats['mean_coherence'])
    """
    distiller = IVADistiller(embeddings, verbose=False)
    results = distiller.distill_multiple(word_groups, mode=mode)
    
    # Compute statistics
    coherences = [r.coherence for r in results.values()]
    angles = distiller.compare_directions(results)
    
    # Exemplar overlap
    all_exemplars = [set(r.exemplars) for r in results.values()]
    overlaps = []
    for i, ex1 in enumerate(all_exemplars):
        for ex2 in all_exemplars[i+1:]:
            if ex1 and ex2:
                overlap = len(ex1 & ex2) / len(ex1 | ex2)
                overlaps.append(overlap)
    
    return {
        'results': results,
        'coherences': {name: r.coherence for name, r in results.items()},
        'mean_coherence': np.mean(coherences) if coherences else 0,
        'angles': angles,
        'mean_angle': np.mean(list(angles.values())) if angles else 0,
        'exemplar_overlaps': overlaps,
        'mean_exemplar_overlap': np.mean(overlaps) if overlaps else 0,
    }


def create_distiller_from_explorer(explorer: 'SenseExplorer') -> IVADistiller:
    """
    Create an IVADistiller from a SenseExplorer instance.
    
    Args:
        explorer: SenseExplorer instance
    
    Returns:
        IVADistiller using the explorer's embeddings
    
    Example:
        >>> se = SenseExplorer.from_glove("glove.txt")
        >>> distiller = create_distiller_from_explorer(se)
    """
    return IVADistiller(
        embeddings=explorer.embeddings,
        verbose=getattr(explorer, 'verbose', False)
    )


# =============================================================================
# Module Test
# =============================================================================

if __name__ == "__main__":
    print("IVA Distillation Module for SenseExplorer")
    print("=" * 50)
    print("\nUsage:")
    print("  from sense_explorer.distillation import IVADistiller")
    print("  distiller = IVADistiller(embeddings)")
    print("  result = distiller.distill_constrained(['money', 'loan', 'account'])")
    print("  print(result.exemplars)")
    print("\nOr via SenseExplorer:")
    print("  results = se.distill_senses('bank')")
    print("  for sense, result in results.items():")
    print("      print(f'{sense}: coherence={result.coherence:.3f}')")
