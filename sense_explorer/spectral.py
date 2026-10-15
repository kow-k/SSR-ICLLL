#!/usr/bin/env python3
"""
Spectral Clustering Module for SenseExplorer
=============================================

Implements spectral clustering with eigengap-based k selection.

Theoretical motivation: If word senses are superposed like waves,
spectral decomposition (eigenvector analysis) should outperform
statistical clustering (BIC-based k selection).

Experimental validation (90% vs 64% at 50d) confirms this prediction.

Author: Kow Kuroda & Claude
"""

import numpy as np
from numpy.linalg import norm, eigh
from typing import Dict, List, Tuple, Optional


def build_similarity_matrix(vectors: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    Build similarity matrix using Gaussian kernel on cosine distances.
    
    Args:
        vectors: (n, d) array of normalized vectors
        sigma: Kernel bandwidth (auto-computed if None)
    
    Returns:
        (n, n) affinity matrix
    """
    n = len(vectors)
    
    # Cosine similarity matrix
    sim_matrix = vectors @ vectors.T
    
    # Convert to distance: d = 1 - sim
    dist_matrix = 1 - sim_matrix
    
    # Auto sigma: median of positive distances
    if sigma is None:
        positive_dists = dist_matrix[dist_matrix > 0]
        sigma = np.median(positive_dists) if len(positive_dists) > 0 else 0.5
        sigma = max(sigma, 0.1)
    
    # Gaussian kernel
    affinity = np.exp(-dist_matrix**2 / (2 * sigma**2))
    np.fill_diagonal(affinity, 0)  # No self-loops
    
    return affinity


def compute_laplacian_eigenvectors(
    affinity: np.ndarray, 
    n_components: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvectors of the normalized graph Laplacian.
    
    Args:
        affinity: (n, n) similarity matrix
        n_components: Number of eigenvectors to compute
    
    Returns:
        (eigenvalues, eigenvectors) sorted by eigenvalue (ascending)
    """
    n = affinity.shape[0]
    
    # Degree matrix
    degrees = affinity.sum(axis=1)
    degrees = np.maximum(degrees, 1e-10)
    
    # Normalized Laplacian: L_sym = I - D^(-1/2) A D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L_sym = np.eye(n) - D_inv_sqrt @ affinity @ D_inv_sqrt
    
    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(L_sym)
    
    # Sort by eigenvalue (should already be sorted, but ensure)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues[:n_components], eigenvectors[:, :n_components]


def find_k_by_eigengap(
    eigenvalues: np.ndarray, 
    min_k: int = 2, 
    max_k: int = 5
) -> int:
    """
    Find optimal k using the eigengap heuristic.
    
    The eigengap method finds the largest "jump" in eigenvalues,
    which indicates the number of well-separated clusters.
    
    Theoretically: The number of eigenvalues close to 0 equals the
    number of connected components. For nearly-disconnected graphs,
    the eigengap reveals how many "almost-components" exist.
    
    Args:
        eigenvalues: Sorted eigenvalues (ascending)
        min_k: Minimum k to consider
        max_k: Maximum k to consider
    
    Returns:
        Optimal k
    """
    # Compute gaps between consecutive eigenvalues
    gaps = np.diff(eigenvalues)
    
    # Look for largest gap in valid range
    # Gap at index i is between eigenvalue[i] and eigenvalue[i+1]
    # If gap is largest at index i, then k = i+1 (eigenvalues 0..i are "connected")
    valid_range = range(min_k - 1, min(max_k, len(gaps)))
    
    if len(list(valid_range)) == 0:
        return min_k
    
    best_gap_idx = max(valid_range, key=lambda i: gaps[i])
    k = best_gap_idx + 1
    
    return max(min_k, min(k, max_k))


def spectral_clustering(
    vectors: np.ndarray, 
    k: int = None, 
    min_k: int = 2, 
    max_k: int = 5,
    sigma: float = None
) -> Tuple[np.ndarray, int]:
    """
    Spectral clustering with optional automatic k selection via eigengap.
    
    Args:
        vectors: (n, d) array of vectors
        k: Number of clusters (if None, determined by eigengap)
        min_k: Minimum k for eigengap search
        max_k: Maximum k for eigengap search
        sigma: Gaussian kernel bandwidth
    
    Returns:
        (labels, k_used)
    """
    n = len(vectors)
    
    if n < min_k:
        return np.zeros(n, dtype=int), 1
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    vectors_norm = vectors / norms
    
    # Build affinity matrix
    affinity = build_similarity_matrix(vectors_norm, sigma)
    
    # Compute Laplacian eigenvectors
    n_components = min(max_k + 2, n)
    eigenvalues, eigenvectors = compute_laplacian_eigenvectors(affinity, n_components)
    
    # Determine k
    if k is None:
        k = find_k_by_eigengap(eigenvalues, min_k, max_k)
    
    k = min(k, n)
    
    # Use first k eigenvectors for clustering
    features = eigenvectors[:, :k]
    
    # Normalize rows (important for spectral clustering)
    row_norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
    features = features / row_norms
    
    # K-means on spectral features
    labels, _ = _kmeans_simple(features, k)
    
    return labels, k


def _kmeans_simple(vectors: np.ndarray, k: int, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means for spectral features."""
    n, d = vectors.shape
    if n < k:
        k = n
    
    # K-means++ initialization
    centroids = np.zeros((k, d))
    centroids[0] = vectors[np.random.randint(n)]
    
    for i in range(1, k):
        dists = np.min([np.sum((vectors - centroids[j])**2, axis=1) for j in range(i)], axis=0)
        dists = np.maximum(dists, 1e-10)
        probs = dists / dists.sum()
        centroids[i] = vectors[np.random.choice(n, p=probs)]
    
    labels = np.zeros(n, dtype=int)
    
    for _ in range(max_iter):
        dists = np.array([np.sum((vectors - c)**2, axis=1) for c in centroids]).T
        new_labels = np.argmin(dists, axis=1)
        
        if np.all(new_labels == labels):
            break
        labels = new_labels
        
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centroids[j] = vectors[mask].mean(axis=0)
    
    return labels, centroids


# =============================================================================
# Integration function for SenseExplorer
# =============================================================================

def discover_anchors_spectral(
    word: str,
    embeddings_norm: Dict[str, np.ndarray],
    embeddings: Dict[str, np.ndarray],
    vocab: set,
    n_anchors: int = 8,
    top_k: int = 50,
    min_senses: int = 2,
    max_senses: int = 5
) -> Tuple[Dict[str, List[str]], int]:
    """
    Discover sense anchors via spectral clustering with eigengap k selection.
    
    This is the recommended method for unsupervised sense discovery,
    achieving 90% accuracy compared to 64% for X-means at 50d.
    
    Theoretical motivation: If senses are superposed like waves,
    spectral decomposition (eigenvector analysis) should identify
    the distinct "frequencies" (senses) in the embedding.
    
    Args:
        word: Target word
        embeddings_norm: Normalized embeddings dict
        embeddings: Original embeddings dict
        vocab: Vocabulary set
        n_anchors: Number of anchors per sense
        top_k: Number of neighbors to consider
        min_senses: Minimum number of senses
        max_senses: Maximum number of senses
    
    Returns:
        Tuple of (anchors dict, discovered_k)
    """
    if word not in vocab:
        return {}, 0
    
    target_emb = embeddings_norm[word]
    
    # Find nearest neighbors
    similarities = []
    for w, emb in embeddings_norm.items():
        if w == word:
            continue
        sim = np.dot(target_emb, emb)
        similarities.append((w, sim))
    
    similarities.sort(key=lambda x: -x[1])
    neighbors = similarities[:top_k]
    
    neighbor_words = [n[0] for n in neighbors]
    neighbor_vecs = np.array([embeddings[w] for w in neighbor_words])
    
    # Spectral clustering with eigengap k selection
    labels, discovered_k = spectral_clustering(
        neighbor_vecs, 
        k=None, 
        min_k=min_senses, 
        max_k=max_senses
    )
    
    # Group neighbors by cluster
    anchors = {}
    for sense_idx in range(discovered_k):
        mask = labels == sense_idx
        cluster_words = [neighbor_words[i] for i, m in enumerate(mask) if m]
        
        if cluster_words:
            anchors[f"sense_{sense_idx}"] = cluster_words[:n_anchors]
    
    return anchors, discovered_k


def discover_anchors_spectral_fixed_k(
    word: str,
    embeddings_norm: Dict[str, np.ndarray],
    embeddings: Dict[str, np.ndarray],
    vocab: set,
    n_senses: int = 2,
    n_anchors: int = 8,
    top_k: int = 50
) -> Dict[str, List[str]]:
    """
    Discover sense anchors via spectral clustering with fixed k.
    
    Use this when you know the number of senses in advance.
    
    Args:
        word: Target word
        embeddings_norm: Normalized embeddings dict
        embeddings: Original embeddings dict
        vocab: Vocabulary set
        n_senses: Number of senses
        n_anchors: Number of anchors per sense
        top_k: Number of neighbors to consider
    
    Returns:
        Anchors dict
    """
    if word not in vocab:
        return {}
    
    target_emb = embeddings_norm[word]
    
    # Find nearest neighbors
    similarities = []
    for w, emb in embeddings_norm.items():
        if w == word:
            continue
        sim = np.dot(target_emb, emb)
        similarities.append((w, sim))
    
    similarities.sort(key=lambda x: -x[1])
    neighbors = similarities[:top_k]
    
    neighbor_words = [n[0] for n in neighbors]
    neighbor_vecs = np.array([embeddings[w] for w in neighbor_words])
    
    # Spectral clustering with fixed k
    labels, _ = spectral_clustering(neighbor_vecs, k=n_senses)
    
    # Group neighbors by cluster
    anchors = {}
    for sense_idx in range(n_senses):
        mask = labels == sense_idx
        cluster_words = [neighbor_words[i] for i, m in enumerate(mask) if m]
        
        if cluster_words:
            anchors[f"sense_{sense_idx}"] = cluster_words[:n_anchors]
    
    return anchors
