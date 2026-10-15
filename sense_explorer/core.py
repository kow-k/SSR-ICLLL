#!/usr/bin/env python3
"""
SenseExplorer: From Sense Discovery to Sense Induction via Simulated Self-Repair
=================================================================================

A lightweight, training-free framework for exploring word sense structure
in static embeddings (GloVe, Word2Vec, FastText, etc.).

Inspired by DNA self-repair mechanisms: noise + self-organization
reveals latent sense structure encoded as stable attractors.

Five operational modes on the supervision continuum:
  - discover_senses_auto(): UNSUPERVISED - geometry decides k and content
  - discover_senses():      SEMI-SUPERVISED - geometry decides content, user decides k
  - separate_senses_wordnet(): WORDNET-GUIDED - lexicographic guidance + geometric filtering
  - induce_senses():        WEAKLY SUPERVISED - anchor-guided induction (88%)
  - merge_with():           CROSS-EMBEDDING - combine sense inventories (NEW in v0.9.3)

The key insight: Senses can be DISCOVERED (emerging from distributional
structure), SEPARATED under lexicographic guidance (WordNet synsets as
structural hints), INDUCED (guided toward anchor-defined targets), or
MERGED across multiple embeddings (combining sense inventories).
All use the same self-repair mechanism but differ in supervision level.

Basic Usage:
  >>> from sense_explorer import SenseExplorer
  >>> se = SenseExplorer.from_glove("glove.6B.300d.txt")
  
  # Unsupervised discovery
  >>> senses = se.discover_senses_auto("bank")
  
  # WordNet-guided separation
  >>> senses = se.separate_senses_wordnet("bank")
  >>> print(senses.keys())  # Synset names as keys
  
  # Weakly supervised induction
  >>> senses = se.induce_senses("bank")
  
  # Cross-embedding merger (NEW in v0.9.3)
  >>> se_twitter = SenseExplorer.from_file("glove.twitter.100d.txt")
  >>> result = se.merge_with(se_twitter, "bank")
  >>> print(f"Convergent: {result.n_convergent}")

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
Repository: https://github.com/kow-k/sense-explorer
"""

__version__ = "0.9.5"
__author__ = "Kow Kuroda & Claude"

import numpy as np
from numpy.linalg import norm
from typing import Dict, List, Tuple, Optional, Union, Set
from pathlib import Path
from collections import defaultdict, Counter
import warnings

# Import hybrid anchor extractor
try:
    from .anchor_extractor import HybridAnchorExtractor, MANUAL_ANCHORS, FRAME_ANCHORS
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Import spectral clustering module
try:
    from .spectral import (
        discover_anchors_spectral, 
        discover_anchors_spectral_fixed_k,
        spectral_clustering
    )
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False

# Import geometry module
try:
    from .geometry import (
        SenseDecomposition,
        decompose as _decompose_geometry,
        print_report as _print_geometry_report,
        print_cross_word_summary as _print_geometry_summary,
        collect_all_angles,
        plot_word_dashboard,
        plot_cross_word_comparison,
        plot_angle_summary,
    )
    GEOMETRY_AVAILABLE = True
except ImportError:
    GEOMETRY_AVAILABLE = False

# Import WordNet for synset-guided separation
try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

# Import embedding merger module (NEW in v0.9.3)
# Located in sense_explorer.merger subpackage
try:
    from .merger import (
        EmbeddingMerger,
        SenseComponent,
        MergerResult,
        create_merger_from_explorers,
        merge_with_ssr,
    )
    MERGER_AVAILABLE = True
except ImportError:
    MERGER_AVAILABLE = False

# Import distillation module (v0.9.2)
try:
    from .distillation import (
        IVADistiller,
        DistillationResult,
        distill_concept,
        measure_set_coherence,
        validate_distillation,
        create_distiller_from_explorer,
    )
    DISTILLATION_AVAILABLE = True
except ImportError:
    DISTILLATION_AVAILABLE = False


# =============================================================================
# Core SenseExplorer Class
# =============================================================================

class SenseExplorer:
    """
    Explore word sense structure via simulated self-repair.
    
    Two operational modes:
    
    1. UNSUPERVISED (discover_senses):
       - True sense discovery from distributional data alone
       - No external knowledge required
       - Senses emerge via k-means clustering of neighbors
       - Achieves ~56% accuracy
    
    2. WEAKLY SUPERVISED (induce_senses):
       - Anchor-guided sense induction
       - Uses FrameNet frames or WordNet glosses as anchors
       - Senses are induced toward anchor-defined targets
       - Achieves ~88% accuracy
    
    Both modes use the same self-repair mechanism:
      1. Creating noisy copies of a word's embedding
      2. Seeding subsets toward different sense directions
      3. Allowing copies to self-organize toward stable configurations
      4. Observing which "attractor basins" the copies settle into
    
    The algorithm is attractor-following, not space-sampling: anchor centroids
    define deterministic attractors, and seeded copies converge to those
    attractors regardless of dimensionality. This means:
      - N (n_copies) can be low (20-50) even for high-dimensional embeddings
      - Anchor quality determines correctness; N only reduces variance
      - Random or poor anchors yield separated but semantically wrong senses
    
    Noise level acts as a granularity control parameter:
      - Low noise (10-20%): Fine distinctions, may over-split
      - Medium noise (30-50%): Standard sense-level distinctions  
      - High noise (60-80%): Coarse groupings
    
    Example:
        >>> se = SenseExplorer.from_glove("glove.6B.100d.txt")
        
        # Unsupervised discovery
        >>> se.discover_senses("bank")
        {'sense_0': array([...]), 'sense_1': array([...])}
        
        # Weakly supervised induction
        >>> se.induce_senses("bank")
        {'financial': array([...]), 'river': array([...])}
    """
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        dim: int = None,
        default_n_senses: int = 2,
        n_copies: int = 30,
        noise_level: float = 0.5,
        seed_strength: float = 0.3,
        n_iterations: int = 15,
        anchor_pull: float = 0.2,
        n_anchors: int = 8,
        top_k: int = 50,
        clustering_method: str = 'spectral',
        use_hybrid_anchors: bool = True,
        verbose: bool = True
    ):
        """
        Initialize SenseExplorer with embeddings.
        
        Args:
            embeddings: Dict mapping words to numpy vectors
            dim: Embedding dimension (auto-detected if None)
            default_n_senses: Default number of senses to discover
            n_copies: Number of noisy copies for self-repair (default: 30).
                         The algorithm is attractor-following, not space-sampling:
                         copies converge to sense attractors defined by anchor
                         centroids. N=20-50 is sufficient regardless of embedding
                         dimensionality. Anchor quality matters far more than N.
            noise_level: Noise magnitude (fraction of embedding values)
                         Acts as granularity control: low=fine, high=coarse
            seed_strength: Strength of initial seeding toward senses
            n_iterations: Number of self-organization iterations
            anchor_pull: Strength of pull toward anchor centroids
            n_anchors: Number of anchors per sense for auto-discovery
            top_k: Number of neighbors for clustering (default: 50)
            clustering_method: Method for unsupervised discovery:
                              'spectral' (default, 90% accuracy at 50d)
                              'xmeans' (BIC-based, 64% accuracy at 50d)
                              'kmeans' (requires n_senses)
            use_hybrid_anchors: Use hybrid anchor extraction (FrameNet + WordNet)
                               Achieves 88% accuracy vs 56% for auto-discovery
            verbose: Print progress messages
        """
        self.embeddings = embeddings
        self.dim = dim or len(next(iter(embeddings.values())))
        self.vocab = set(embeddings.keys())
        self.vocab_size = len(self.vocab)
        
        # Self-repair parameters
        self.default_n_senses = default_n_senses
        self.n_copies = n_copies
        self.noise_level = noise_level
        self.seed_strength = seed_strength
        self.n_iterations = n_iterations
        self.anchor_pull = anchor_pull
        self.n_anchors = n_anchors
        self.top_k = top_k
        self.clustering_method = clustering_method
        self.use_hybrid_anchors = use_hybrid_anchors
        self.verbose = verbose
        
        # Validate clustering method
        valid_methods = ['spectral', 'xmeans', 'kmeans']
        if clustering_method not in valid_methods:
            raise ValueError(f"clustering_method must be one of {valid_methods}")
        
        if clustering_method == 'spectral' and not SPECTRAL_AVAILABLE:
            warnings.warn("Spectral clustering not available, falling back to xmeans")
            self.clustering_method = 'xmeans'
        
        # Initialize hybrid anchor extractor if available and enabled
        self._hybrid_extractor = None
        if use_hybrid_anchors and HYBRID_AVAILABLE:
            self._hybrid_extractor = HybridAnchorExtractor(
                self.vocab,
                use_manual=True,
                use_framenet=True,
                use_wordnet=True,
                verbose=verbose
            )
        
        # Caches
        self._sense_cache = {}  # word -> {sense_name: embedding}
        self._anchor_cache = {}  # word -> {sense_name: [anchor_words]}
        self._stability_cache = {}  # word -> stability analysis results
        
        # Precompute normalized embeddings for speed
        self._embeddings_norm = {}
        for word, emb in embeddings.items():
            self._embeddings_norm[word] = emb / (norm(emb) + 1e-10)
        
        if verbose:
            hybrid_status = "enabled" if (use_hybrid_anchors and HYBRID_AVAILABLE) else "disabled"
            print(f"SenseExplorer v{__version__} initialized with {self.vocab_size:,} words, dim={self.dim}")
            print(f"  Clustering method: {self.clustering_method} (top_k={self.top_k})")
            print(f"  Hybrid anchor extraction: {hybrid_status}")
            if use_hybrid_anchors and HYBRID_AVAILABLE:
                print(f"  Coverage: manual={len(MANUAL_ANCHORS)} words, framenet={len(FRAME_ANCHORS)} frames")
    
    # =========================================================================
    # Loading Methods
    # =========================================================================
    
    @classmethod
    def from_glove(cls, filepath: str, max_words: int = None, **kwargs) -> 'SenseExplorer':
        """
        Load from GloVe format (text or binary).
        
        Args:
            filepath: Path to GloVe file (.txt or .bin)
            max_words: Maximum number of words to load
            **kwargs: Additional arguments for SenseExplorer
        
        Returns:
            SenseExplorer instance
        """
        embeddings, dim = cls._load_glove(filepath, max_words, kwargs.get('verbose', True))
        return cls(embeddings, dim=dim, **kwargs)
    
    @classmethod
    def from_word2vec(cls, filepath: str, max_words: int = None, binary: bool = True, **kwargs) -> 'SenseExplorer':
        """
        Load from Word2Vec format.
        
        Args:
            filepath: Path to Word2Vec file
            max_words: Maximum number of words to load
            binary: Whether file is binary format
            **kwargs: Additional arguments for SenseExplorer
        
        Returns:
            SenseExplorer instance
        """
        embeddings, dim = cls._load_word2vec(filepath, max_words, binary, kwargs.get('verbose', True))
        return cls(embeddings, dim=dim, **kwargs)
    
    @classmethod
    def from_dict(cls, embeddings: Dict[str, Union[np.ndarray, List[float]]], **kwargs) -> 'SenseExplorer':
        """
        Create from dictionary of embeddings.
        
        Args:
            embeddings: Dict mapping words to vectors (numpy arrays or lists)
            **kwargs: Additional arguments for SenseExplorer
        
        Returns:
            SenseExplorer instance
        """
        # Convert lists to numpy arrays if needed
        emb_dict = {}
        for word, vec in embeddings.items():
            if isinstance(vec, list):
                emb_dict[word] = np.array(vec, dtype=np.float32)
            else:
                emb_dict[word] = vec.astype(np.float32)
        
        return cls(emb_dict, **kwargs)
    
    @classmethod
    def from_file(cls, filepath: str, max_words: int = None, target_dim: int = None, **kwargs) -> 'SenseExplorer':
        """
        Auto-detect format and load embeddings.

        Supported formats:
          - .txt: GloVe text format
          - .txt.gz / .gz: Gzipped GloVe text format
          - .bin: Gensim KeyedVectors or Word2Vec binary
          - .model: Gensim native saved model

        Args:
            filepath: Path to embedding file
            max_words: Maximum number of words to load
            target_dim: Target dimension for alignment. If provided, embeddings
                       will be truncated or zero-padded to this dimension.
                       Useful for merging embeddings with different dimensions.
            **kwargs: Additional arguments for SenseExplorer

        Returns:
            SenseExplorer instance
        """
        import os
        ext = os.path.splitext(filepath)[1].lower()
        verbose = kwargs.get('verbose', True)

        if ext == '.gz':
            # Peek to determine if gzipped text or gzipped binary
            import gzip as _gzip
            with _gzip.open(filepath, 'rb') as f:
                sample = f.read(100)
            try:
                sample.decode('utf-8')
                # Valid UTF-8 — gzipped GloVe text
                embeddings, dim = cls._load_glove(filepath, max_words, verbose)
            except UnicodeDecodeError:
                # Binary data — gzipped word2vec binary
                embeddings, dim = cls._load_gensim_model(filepath, max_words, verbose)
        elif ext == '.model':
            # Gensim native format
            embeddings, dim = cls._load_gensim_model(filepath, max_words, verbose)
        elif ext == '.bin':
            # Try Gensim/Word2Vec binary first, fall back to GloVe binary
            try:
                embeddings, dim = cls._load_gensim_model(filepath, max_words, verbose)
            except Exception:
                embeddings, dim = cls._load_glove(filepath, max_words, verbose)
        elif ext == '.npy':
            raise ValueError(
                f"Cannot load .npy directly. Use the .model file instead: "
                f"{filepath.replace('.vectors.npy', '')}")
        else:
            # Default: GloVe text
            embeddings, dim = cls._load_glove(filepath, max_words, verbose)
        
        # Verify and fix dimensions
        # Some files have inconsistent dimensions or misleading headers
        if embeddings and target_dim is not None:
            # Check all vectors and align any that don't match target_dim
            needs_alignment = False
            for word, vec in embeddings.items():
                if len(vec) != target_dim:
                    needs_alignment = True
                    break
            
            if needs_alignment:
                # Find the most common dimension
                sample_dims = [len(v) for v in list(embeddings.values())[:100]]
                actual_dim = max(set(sample_dims), key=sample_dims.count)
                if verbose:
                    print(f"  Detected actual dimension: {actual_dim}d (target: {target_dim}d)")
                embeddings, dim = cls._align_dimensions(embeddings, actual_dim, target_dim, verbose)
        elif embeddings:
            # No target_dim specified, but verify consistency
            sample_vec = next(iter(embeddings.values()))
            actual_dim = len(sample_vec)
            if actual_dim != dim:
                if verbose:
                    print(f"  Warning: Reported dim={dim} but actual dim={actual_dim}")
                dim = actual_dim
        
        return cls(embeddings, dim=dim, **kwargs)
    
    @staticmethod
    def _align_dimensions(
        embeddings: Dict[str, np.ndarray],
        current_dim: int,
        target_dim: int,
        verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Align embedding dimensions to target.
        
        Args:
            embeddings: Dict mapping words to vectors
            current_dim: Most common current dimension of embeddings
            target_dim: Target dimension
            verbose: Print progress
            
        Returns:
            Tuple of (aligned_embeddings, target_dim)
        """
        if verbose:
            if current_dim > target_dim:
                print(f"  Truncating embeddings from {current_dim}d to {target_dim}d")
            elif current_dim < target_dim:
                print(f"  Padding embeddings from {current_dim}d to {target_dim}d")
            else:
                print(f"  Aligning embeddings to {target_dim}d (fixing inconsistent dimensions)")
        
        aligned = {}
        for word, vec in embeddings.items():
            vec_dim = len(vec)
            if vec_dim == target_dim:
                aligned[word] = vec
            elif vec_dim > target_dim:
                # Truncate
                aligned[word] = vec[:target_dim].copy()
            else:
                # Pad with zeros
                padded = np.zeros(target_dim, dtype=vec.dtype)
                padded[:vec_dim] = vec
                aligned[word] = padded
        
        return aligned, target_dim

    @staticmethod
    def _load_gensim_model(filepath: str, max_words: int = None,
                           verbose: bool = True) -> Tuple[Dict[str, np.ndarray], int]:
        """Load embeddings from Gensim KeyedVectors (.model or .bin)."""
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError(
                "gensim is required for loading .model/.bin files. "
                "Install with: pip install gensim")

        if verbose:
            print(f"Loading Gensim embeddings from {filepath}...")

        # Try native Gensim format, then Word2Vec binary
        try:
            kv = KeyedVectors.load(filepath)
            if verbose:
                print(f"  Loaded as native Gensim format")
        except Exception:
            try:
                kv = KeyedVectors.load_word2vec_format(filepath, binary=True)
                if verbose:
                    print(f"  Loaded as Word2Vec binary format")
            except Exception:
                kv = KeyedVectors.load(filepath, mmap='r')
                if verbose:
                    print(f"  Loaded as native Gensim format (mmap)")

        vocab = kv.index_to_key
        if max_words:
            vocab = vocab[:max_words]

        embeddings = {}
        for word in vocab:
            embeddings[word] = kv[word].astype(np.float32)

        dim = kv.vector_size
        if verbose:
            print(f"  Loaded {len(embeddings):,} embeddings with dimension {dim}")
        return embeddings, dim

    @staticmethod
    def _load_glove(filepath: str, max_words: int = None, verbose: bool = True) -> Tuple[Dict[str, np.ndarray], int]:
        """Load GloVe embeddings from file (text, binary, or gzipped text)."""
        import gzip
        embeddings = {}
        dim = None
        
        if verbose:
            print(f"Loading embeddings from {filepath}...")
        
        filepath = Path(filepath)
        
        if filepath.suffix == '.bin':
            # Binary format (gensim-style)
            with open(filepath, 'rb') as f:
                header = f.readline().decode('utf-8').strip()
                vocab_size, dim = map(int, header.split())
                
                for _ in range(vocab_size):
                    word_bytes = b''
                    while True:
                        c = f.read(1)
                        if c == b' ' or c == b'':
                            break
                        word_bytes += c
                    word = word_bytes.decode('utf-8', errors='replace')
                    vec = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
                    embeddings[word] = vec
                    
                    if max_words and len(embeddings) >= max_words:
                        break
        elif filepath.suffix == '.gz':
            # Gzipped text format (may have word2vec-style header)
            first_line = True
            with gzip.open(filepath, 'rt', encoding='utf-8', errors='replace') as f:
                for line in f:
                    parts = line.strip().split()
                    # Detect word2vec-style header: exactly 2 integer tokens
                    if first_line and len(parts) == 2:
                        first_line = False
                        try:
                            int(parts[0])
                            int(parts[1])
                            if verbose:
                                print(f"  Detected word2vec header: {parts[0]} words, "
                                      f"{parts[1]} dimensions — skipping")
                            continue
                        except ValueError:
                            pass  # Not a header
                    first_line = False
                    if len(parts) < 3:
                        continue
                    word = parts[0]
                    try:
                        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        embeddings[word] = vec
                        if dim is None:
                            dim = len(vec)
                        if max_words and len(embeddings) >= max_words:
                            break
                    except ValueError:
                        continue
        else:
            # Text format (handles both raw GloVe and word2vec-style headers)
            first_line = True
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    parts = line.strip().split()
                    # Detect word2vec-style header: exactly 2 integer tokens
                    if first_line and len(parts) == 2:
                        first_line = False
                        try:
                            int(parts[0])
                            int(parts[1])
                            if verbose:
                                print(f"  Detected word2vec header: {parts[0]} words, "
                                      f"{parts[1]} dimensions — skipping")
                            continue
                        except ValueError:
                            pass  # Not a header; fall through to normal parsing
                    first_line = False
                    if len(parts) < 2:
                        continue
                    word = parts[0]
                    try:
                        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        embeddings[word] = vec
                        if dim is None:
                            dim = len(vec)
                        if max_words and len(embeddings) >= max_words:
                            break
                    except ValueError:
                        continue
        
        if verbose:
            print(f"Loaded {len(embeddings):,} embeddings with dimension {dim}")
        
        return embeddings, dim
    
    @staticmethod
    def _load_word2vec(filepath: str, max_words: int = None, binary: bool = True, verbose: bool = True) -> Tuple[Dict[str, np.ndarray], int]:
        """Load Word2Vec embeddings from file."""
        embeddings = {}
        
        if verbose:
            print(f"Loading Word2Vec embeddings from {filepath}...")
        
        with open(filepath, 'rb') as f:
            header = f.readline().decode('utf-8').strip()
            vocab_size, dim = map(int, header.split())
            
            for _ in range(vocab_size):
                word_bytes = b''
                while True:
                    c = f.read(1)
                    if c == b' ':
                        break
                    if c == b'':
                        break
                    word_bytes += c
                
                word = word_bytes.decode('utf-8', errors='replace')
                
                if binary:
                    vec = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
                else:
                    vec = np.array([float(x) for x in f.readline().split()], dtype=np.float32)
                
                embeddings[word] = vec
                
                if max_words and len(embeddings) >= max_words:
                    break
        
        if verbose:
            print(f"Loaded {len(embeddings):,} embeddings with dimension {dim}")
        
        return embeddings, dim
    
    # =========================================================================
    # Core Sense Discovery
    # =========================================================================
    
    def discover_senses(
        self,
        word: str,
        n_senses: int = None,
        noise_level: float = None,
        clustering_method: str = None,
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        UNSUPERVISED sense discovery via clustering.
        
        This is true sense discovery: senses emerge from the distributional
        structure alone, with no external knowledge or anchor guidance.
        
        With spectral clustering (default): achieves ~90% accuracy at 50d
        With X-means: achieves ~64% accuracy at 50d
        
        Args:
            word: Target word
            n_senses: Number of senses to discover (default: 2)
                     Only used if clustering_method='kmeans'
            noise_level: Override default noise level (granularity control)
            clustering_method: Override default clustering method
                              ('spectral', 'xmeans', 'kmeans')
            force: Force rediscovery even if cached
        
        Returns:
            Dict mapping sense names (sense_0, sense_1, ...) to embeddings
        
        Example:
            >>> se = SenseExplorer.from_glove("glove.txt")
            >>> senses = se.discover_senses("bank", n_senses=2)
            >>> print(senses.keys())
            dict_keys(['sense_0', 'sense_1'])
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        cache_key = f"{word}_discovered"
        if not force and cache_key in self._sense_cache:
            return self._sense_cache[cache_key]
        
        n_senses = n_senses or self.default_n_senses
        noise = noise_level if noise_level is not None else self.noise_level
        method = clustering_method or self.clustering_method
        
        # Select clustering method
        if method == 'spectral' and SPECTRAL_AVAILABLE:
            anchors = discover_anchors_spectral_fixed_k(
                word, 
                self._embeddings_norm, 
                self.embeddings,
                self.vocab,
                n_senses=n_senses,
                n_anchors=self.n_anchors,
                top_k=self.top_k
            )
        elif method == 'xmeans':
            anchors, _ = self._discover_anchors_xmeans(word)
        else:  # kmeans
            anchors = self._discover_anchors_kmeans(word, n_senses)
        
        sense_centroids = self._compute_sense_centroids(anchors)
        
        if len(sense_centroids) < 2:
            emb = self._embeddings_norm[word]
            self._sense_cache[cache_key] = {'default': emb}
            return {'default': emb}
        
        sense_embs = self._simulated_repair(word, sense_centroids, noise, sense_loyal=False)
        
        self._sense_cache[cache_key] = sense_embs
        self._anchor_cache[cache_key] = anchors
        
        if self.verbose:
            print(f"  [UNSUPERVISED/{method}] Discovered {len(sense_embs)} senses for '{word}'")
        
        return sense_embs
    
    def induce_senses(
        self,
        word: str,
        anchors: Dict[str, List[str]] = None,
        n_senses: int = None,
        noise_level: float = None,
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        WEAKLY SUPERVISED sense induction via anchor guidance.
        
        Senses are induced toward targets defined by anchor words.
        Uses hybrid extraction (FrameNet + WordNet) when anchors not provided.
        Achieves ~88% accuracy with frame-based anchors.
        
        Args:
            word: Target word
            anchors: Optional dict of {sense_name: [anchor_words]}
                     If None, uses hybrid extraction (FrameNet → WordNet → k-means)
            n_senses: Number of senses (used only if anchors is None)
            noise_level: Override default noise level (granularity control)
            force: Force re-induction even if cached
        
        Returns:
            Dict mapping sense names to sense-specific embeddings
        
        Example:
            >>> se = SenseExplorer.from_glove("glove.txt")
            >>> senses = se.induce_senses("bank")
            >>> print(senses.keys())
            dict_keys(['financial', 'river'])
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        cache_key = f"{word}_induced"
        if not force and cache_key in self._sense_cache:
            return self._sense_cache[cache_key]
        
        n_senses = n_senses or self.default_n_senses
        noise = noise_level if noise_level is not None else self.noise_level
        
        # Weakly supervised: use provided anchors or hybrid extraction
        if anchors is None:
            anchors = self._extract_anchors_hybrid(word, n_senses)
        
        # Validate anchor quality (warns if poor)
        anchor_quality = self._validate_anchors(word, anchors, warn=self.verbose)
        
        sense_centroids = self._compute_sense_centroids(anchors)
        
        if len(sense_centroids) < 2:
            emb = self._embeddings_norm[word]
            self._sense_cache[cache_key] = {'default': emb}
            return {'default': emb}
        
        sense_embs = self._simulated_repair(word, sense_centroids, noise, sense_loyal=True)
        
        self._sense_cache[cache_key] = sense_embs
        self._anchor_cache[cache_key] = anchors
        self._anchor_quality_cache = getattr(self, '_anchor_quality_cache', {})
        self._anchor_quality_cache[cache_key] = anchor_quality
        
        if self.verbose:
            quality_summary = {s: info['quality'] for s, info in anchor_quality.items()}
            print(f"  [WEAKLY SUPERVISED] Induced {len(sense_embs)} senses for '{word}'")
            print(f"  Anchor quality: {quality_summary}")
        
        return sense_embs

    # =========================================================================
    # WordNet-Guided Sense Separation
    # =========================================================================

    def separate_senses_wordnet(
        self,
        word: str,
        hyponym_depth: int = 2,
        merge_threshold: float = 0.70,
        min_anchors: int = 2,
        pos_filter: str = None,
        noise_level: float = None,
        force: bool = False,
        return_details: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        WORDNET-GUIDED sense separation via synset-derived anchors.

        Bridges unsupervised discovery and supervised induction: WordNet
        provides structural guidance (which senses to look for), while
        embedding geometry determines which senses the corpus actually
        supports. The same attractor-following mechanism as discover_senses
        and induce_senses, but with anchors derived automatically from
        WordNet's sense inventory.

        The method operates on the supervision continuum:
          - discover_senses_auto: fully unsupervised (geometry decides k and content)
          - discover_senses:      geometry decides content, user decides k
          - separate_senses_wordnet: WordNet guides both k and content (THIS)
          - induce_senses:        user provides anchors directly

        Algorithm:
          1. Query WordNet for all synsets of the target word
          2. For each synset, collect lemmas + hyponym lemmas as anchors
          3. Filter for embedding vocabulary presence
          4. Merge synsets whose anchor centroids overlap in embedding space
             (cosine similarity > merge_threshold)
          5. Feed merged anchor groups into attractor-following (sense_loyal)
          6. Return sense vectors keyed by synset names

        Args:
            word: Target polysemous word
            hyponym_depth: Levels of hyponyms to traverse (default: 2)
            merge_threshold: Cosine similarity above which synset groups
                are merged (default: 0.70). Lower = more aggressive merging.
            min_anchors: Minimum in-vocabulary anchors for a synset to be
                viable (default: 2). Synsets with fewer are dropped.
            pos_filter: Restrict to POS ('n', 'v', 'a', 'r') or None for all.
            noise_level: Override default noise level.
            force: Force re-separation even if cached.
            return_details: If True, return (sense_embs, details_dict) with
                anchor groups, merge history, and synset metadata.

        Returns:
            Dict mapping sense names (synset names) to sense-specific embeddings.
            If return_details=True, returns (sense_embs, details) tuple.

        Raises:
            ImportError: If NLTK/WordNet is not installed.
            ValueError: If word is not in vocabulary.

        Example:
            >>> se = SenseExplorer.from_glove("glove.6B.300d.txt")
            >>> senses = se.separate_senses_wordnet("bank")
            >>> print(senses.keys())
            dict_keys(['depository_financial_institution.n.01', 'bank.n.01'])
        """
        if not WORDNET_AVAILABLE:
            raise ImportError(
                "WordNet-guided separation requires NLTK with WordNet data. "
                "Install with: pip install nltk && python -c "
                "\"import nltk; nltk.download('wordnet')\""
            )

        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")

        cache_key = f"{word}_wordnet"
        if not force and cache_key in self._sense_cache:
            if return_details:
                details = getattr(self, '_wordnet_details_cache', {}).get(cache_key, {})
                return self._sense_cache[cache_key], details
            return self._sense_cache[cache_key]

        noise = noise_level if noise_level is not None else self.noise_level

        # Step 1: Get synsets
        synsets = wn.synsets(word)
        if pos_filter:
            pos_map = {'n': wn.NOUN, 'v': wn.VERB, 'a': wn.ADJ, 'r': wn.ADV}
            wn_pos = pos_map.get(pos_filter)
            if wn_pos:
                synsets = [s for s in synsets if s.pos() == wn_pos]

        if not synsets:
            emb = self._embeddings_norm[word]
            result = {'default': emb}
            self._sense_cache[cache_key] = result
            if return_details:
                return result, {'synsets': [], 'reason': 'no_synsets'}
            return result

        # Step 2: Extract anchors per synset
        synset_anchors = self._extract_wordnet_anchors(
            word, synsets, hyponym_depth, min_anchors)

        if self.verbose:
            print(f"  [WORDNET] {len(synsets)} synsets for '{word}', "
                  f"{len(synset_anchors)} have ≥{min_anchors} in-vocab anchors")

        if len(synset_anchors) < 2:
            emb = self._embeddings_norm[word]
            result = {'default': emb}
            self._sense_cache[cache_key] = result
            if return_details:
                return result, {
                    'synsets': synsets,
                    'synset_anchors': synset_anchors,
                    'reason': 'insufficient_viable_synsets'
                }
            return result

        # Step 3: Merge synsets with overlapping anchor centroids
        merged_groups, merge_history = self._merge_synset_groups(
            synset_anchors, merge_threshold)

        if self.verbose:
            print(f"  [WORDNET] Merged {len(synset_anchors)} synset groups "
                  f"→ {len(merged_groups)} distinct sense groups")
            for name, anchors in merged_groups.items():
                print(f"    {name}: {len(anchors)} anchors "
                      f"({', '.join(list(anchors)[:5])}{'...' if len(anchors) > 5 else ''})")

        if len(merged_groups) < 2:
            emb = self._embeddings_norm[word]
            result = {'default': emb}
            self._sense_cache[cache_key] = result
            if return_details:
                return result, {
                    'synsets': synsets,
                    'merged_groups': merged_groups,
                    'merge_history': merge_history,
                    'reason': 'all_synsets_merged'
                }
            return result

        # Step 4: Validate anchors
        anchor_quality = self._validate_anchors(word, merged_groups, warn=self.verbose)

        # Step 5: Compute centroids and run attractor-following
        sense_centroids = self._compute_sense_centroids(merged_groups)

        if len(sense_centroids) < 2:
            emb = self._embeddings_norm[word]
            result = {'default': emb}
            self._sense_cache[cache_key] = result
            if return_details:
                return result, {
                    'synsets': synsets,
                    'reason': 'centroids_collapsed'
                }
            return result

        sense_embs = self._simulated_repair(
            word, sense_centroids, noise, sense_loyal=True)

        # Cache results
        self._sense_cache[cache_key] = sense_embs
        self._anchor_cache[cache_key] = merged_groups

        details = {
            'synsets': synsets,
            'n_synsets_total': len(synsets),
            'synset_anchors': synset_anchors,
            'merged_groups': merged_groups,
            'merge_history': merge_history,
            'anchor_quality': anchor_quality,
            'n_groups_after_merge': len(merged_groups),
            'n_senses_returned': len(sense_embs),
        }
        if not hasattr(self, '_wordnet_details_cache'):
            self._wordnet_details_cache = {}
        self._wordnet_details_cache[cache_key] = details

        if self.verbose:
            quality_summary = {s: info['quality'] for s, info in anchor_quality.items()}
            print(f"  [WORDNET] Separated {len(sense_embs)} senses for '{word}'")
            print(f"  Anchor quality: {quality_summary}")

        if return_details:
            return sense_embs, details
        return sense_embs

    def _extract_wordnet_anchors(
        self,
        word: str,
        synsets: list,
        hyponym_depth: int = 2,
        min_anchors: int = 2,
    ) -> Dict[str, List[str]]:
        """Extract vocabulary-filtered anchor words from WordNet synsets.

        For each synset, collects lemma names and hyponym lemma names
        (to specified depth), filters for embedding vocabulary presence,
        and removes the target word itself.

        Args:
            word: Target word (excluded from anchor lists).
            synsets: List of WordNet synset objects.
            hyponym_depth: Levels of hyponym tree to traverse.
            min_anchors: Minimum in-vocabulary anchors for viability.

        Returns:
            Dict of {synset_name: [anchor_words]} for viable synsets.
        """
        result = {}
        target_lemmas = {word, word.lower(), word.upper()}

        for synset in synsets:
            anchors = set()

            # Direct lemma names
            for lemma in synset.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                for part in name.split():
                    anchors.add(part)

            # Hypernyms (1 level)
            for hyper in synset.hypernyms():
                for lemma in hyper.lemmas():
                    name = lemma.name().lower().replace('_', ' ')
                    for part in name.split():
                        anchors.add(part)

            # Hyponyms (recursive)
            def collect_hypo(ss, depth):
                if depth <= 0:
                    return
                for hypo in ss.hyponyms():
                    for lemma in hypo.lemmas():
                        name = lemma.name().lower().replace('_', ' ')
                        for part in name.split():
                            anchors.add(part)
                    collect_hypo(hypo, depth - 1)

            collect_hypo(synset, hyponym_depth)

            # Similar-to and also-see
            for related in synset.similar_tos() + synset.also_sees():
                for lemma in related.lemmas():
                    name = lemma.name().lower().replace('_', ' ')
                    for part in name.split():
                        anchors.add(part)

            # Gloss words (filtered)
            _stopwords = {
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
                'whom', 'what', 'where', 'when', 'how', 'all', 'each',
                'every', 'both', 'few', 'more', 'most', 'other', 'some',
                'any', 'etc',
            }
            for text in [synset.definition()] + synset.examples():
                for w in text.split():
                    cleaned = w.strip('.,;:!?()[]"\'').lower()
                    if len(cleaned) > 2 and cleaned not in _stopwords:
                        anchors.add(cleaned)

            # Filter: in vocabulary, not the target word
            anchors = [a for a in anchors
                       if a in self.vocab and a not in target_lemmas]

            if len(anchors) >= min_anchors:
                result[synset.name()] = anchors

        return result

    def _merge_synset_groups(
        self,
        synset_anchors: Dict[str, List[str]],
        merge_threshold: float = 0.70,
    ) -> tuple:
        """Merge synsets whose anchor centroids are too similar.

        Iteratively merges the most similar pair until no pair exceeds
        the threshold. Merged group names join with '+'.

        Args:
            synset_anchors: Dict of {synset_name: [anchor_words]}.
            merge_threshold: Cosine similarity above which groups merge.

        Returns:
            Tuple of (merged_groups, merge_history) where merged_groups
            is Dict[str, List[str]] and merge_history is a list of
            (group1, group2, similarity) tuples.
        """
        # Compute centroids for each group
        groups = dict(synset_anchors)  # copy
        merge_history = []

        while len(groups) > 1:
            # Compute centroids
            centroids = {}
            for name, anchors in groups.items():
                vecs = [self._embeddings_norm[w] for w in anchors if w in self.vocab]
                if vecs:
                    c = np.mean(vecs, axis=0)
                    c = c / (norm(c) + 1e-10)
                    centroids[name] = c

            # Find most similar pair
            names = list(centroids.keys())
            best_sim = -1
            best_pair = None

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    sim = float(centroids[names[i]] @ centroids[names[j]])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (names[i], names[j])

            if best_sim < merge_threshold or best_pair is None:
                break

            # Merge
            n1, n2 = best_pair
            merged_name = f"{n1}+{n2}"
            # Deduplicate anchors
            merged_anchors = list(set(groups[n1] + groups[n2]))
            groups[merged_name] = merged_anchors
            del groups[n1]
            del groups[n2]
            merge_history.append((n1, n2, best_sim))

            if self.verbose:
                print(f"    Merged {n1} + {n2} (cos={best_sim:.3f})")

        return groups, merge_history

    @staticmethod
    def wordnet_sweep_k_values(n_synsets: int) -> List[int]:
        """Compute k values for WordNet-guided granularity sweep.

        Given N synsets, returns sorted unique k values from
        k = round(N/i) for i in [1, 1.5, 2, 2.5, 3, 4, 5, ...], descending.
        The half-steps at i=1.5 and i=2.5 fill important gaps between
        the coarsest levels (e.g., N=18: k=12 between 18 and 9).
        Filters out k < 2.

        Args:
            n_synsets: Number of WordNet synsets for the target word.

        Returns:
            List of k values in descending order.

        Example:
            >>> SenseExplorer.wordnet_sweep_k_values(18)
            [18, 12, 9, 7, 6, 4, 3, 2]
        """
        max_i = max(1, round(n_synsets / 2))
        divisors = [1, 1.5, 2, 2.5] + list(range(3, max_i + 1))
        divisors = [d for d in divisors if d <= max_i]
        k_values = set()
        for d in divisors:
            k = round(n_synsets / d)
            if k >= 2:
                k_values.add(k)
        return sorted(k_values, reverse=True)

    def sweep_senses_wordnet(
        self,
        word: str,
        k_values: List[int] = None,
        **kwargs,
    ) -> List[Dict]:
        """Sweep sense separation across WordNet-guided granularity levels.

        Runs discover_senses at each k value derived from the WordNet
        synset count, providing a top-down view from lexicographic
        granularity to the coarsest separation the corpus supports.

        This is the SenseExplorer-native version of the --wordnet-sweep
        functionality in run_synset_mapping.py.

        Args:
            word: Target polysemous word.
            k_values: Explicit list of k values to sweep. If None,
                auto-computed from WordNet synset count via
                wordnet_sweep_k_values().
            **kwargs: Passed to discover_senses (e.g., noise_level).

        Returns:
            List of dicts, one per k level (descending), each containing:
              - 'requested_k': The k value requested
              - 'actual_k': Number of senses actually returned
              - 'senses': Dict[str, np.ndarray] of sense embeddings
              - 'inter_sense_angles': List of (s1, s2, angle_deg) tuples

        Raises:
            ImportError: If NLTK/WordNet is not installed.
            ValueError: If word is not in vocabulary.

        Example:
            >>> se = SenseExplorer.from_glove("glove.6B.300d.txt")
            >>> results = se.sweep_senses_wordnet("bank")
            >>> for r in results:
            ...     print(f"k={r['requested_k']}→{r['actual_k']}, "
            ...           f"min_angle={min(a for _,_,a in r['inter_sense_angles']) if r['inter_sense_angles'] else float('nan'):.1f}°")
        """
        if not WORDNET_AVAILABLE:
            raise ImportError(
                "WordNet sweep requires NLTK with WordNet data. "
                "Install with: pip install nltk && python -c "
                "\"import nltk; nltk.download('wordnet')\""
            )

        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")

        # Determine k values
        if k_values is None:
            synsets = wn.synsets(word)
            n_synsets = len(synsets)
            if n_synsets < 2:
                if self.verbose:
                    print(f"  [SWEEP] '{word}' has {n_synsets} synset(s), nothing to sweep")
                return []
            k_values = self.wordnet_sweep_k_values(n_synsets)
            if self.verbose:
                print(f"  [SWEEP] '{word}': N={n_synsets} synsets → "
                      f"k values: {k_values}")

        results = []
        for target_k in k_values:
            try:
                sense_dict = self.discover_senses(
                    word, n_senses=target_k, force=True, **kwargs)
                sense_names = sorted(sense_dict.keys())
                actual_k = len(sense_names)

                # Compute inter-sense angles
                angles = []
                vecs = [sense_dict[s] for s in sense_names]
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        cos_sim = float(vecs[i] @ vecs[j])
                        cos_sim = max(-1.0, min(1.0, cos_sim))
                        angle_deg = float(np.degrees(np.arccos(cos_sim)))
                        angles.append((sense_names[i], sense_names[j], angle_deg))

                result = {
                    'requested_k': target_k,
                    'actual_k': actual_k,
                    'senses': sense_dict,
                    'inter_sense_angles': angles,
                }
                results.append(result)

                if self.verbose:
                    min_angle = min(a for _, _, a in angles) if angles else float('nan')
                    k_str = f"{target_k}" if actual_k == target_k else f"{target_k}→{actual_k}"
                    print(f"    k={k_str}: {actual_k} senses, "
                          f"min angle {min_angle:.1f}°")

            except Exception as e:
                if self.verbose:
                    print(f"    k={target_k}: failed ({e})")

        return results

    def explore_senses(
        self,
        word: str,
        mode: str = 'auto',
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method: explore senses using specified mode.
        
        Args:
            word: Target word
            mode: One of:
                  - 'discover': Unsupervised with specified n_senses
                  - 'discover_auto': Unsupervised, auto-detect n_senses
                  - 'induce': Weakly supervised with anchors
                  - 'wordnet': WordNet-guided separation
                  - 'auto': Induce if anchors available, else discover_auto
            **kwargs: Passed to the underlying method
        
        Returns:
            Dict mapping sense names to sense-specific embeddings
        
        Example:
            >>> se.explore_senses("bank", mode='discover_auto')  # Spectral
            >>> se.explore_senses("bank", mode='induce')         # With anchors
            >>> se.explore_senses("bank", mode='wordnet')        # WordNet-guided
            >>> se.explore_senses("bank", mode='auto')           # Best available
        """
        if mode == 'discover':
            return self.discover_senses(word, **kwargs)
        elif mode == 'discover_auto':
            return self.discover_senses_auto(word, **kwargs)
        elif mode == 'induce':
            return self.induce_senses(word, **kwargs)
        elif mode == 'wordnet':
            return self.separate_senses_wordnet(word, **kwargs)
        elif mode == 'auto':
            # Try hybrid extraction; if fails, fall back to X-means discovery
            if self._hybrid_extractor is not None:
                anchors, source = self._hybrid_extractor.extract(word, kwargs.get('n_senses', 2))
                if anchors and source != 'auto':
                    return self.induce_senses(word, anchors=anchors, **kwargs)
            return self.discover_senses_auto(word, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'discover', "
                           f"'discover_auto', 'induce', 'wordnet', or 'auto'")
    
    def _extract_anchors_hybrid(self, word: str, n_senses: int) -> Dict[str, List[str]]:
        """
        Extract anchors using hybrid strategy (FrameNet → WordNet → k-means).
        
        This is for weakly supervised induction.
        """
        if self._hybrid_extractor is not None:
            anchors, source = self._hybrid_extractor.extract(word, n_senses)
            if anchors:
                if self.verbose:
                    print(f"  Anchors extracted via {source} for '{word}'")
                return anchors
        
        # Fall back to k-means if hybrid fails
        if self.verbose:
            print(f"  Falling back to k-means for '{word}'")
        return self._discover_anchors_kmeans(word, n_senses)

    # =========================================================================
    # Sense Geometry Analysis
    # =========================================================================

    def localize_senses(
        self,
        word: str,
        senses: Dict[str, np.ndarray] = None,
        mode: str = 'induce',
        force: bool = False,
        **kwargs
    ) -> 'SenseDecomposition':
        """
        Decompose a word vector into its sense components and analyze
        the geometric relationships between senses.

        This performs the linear decomposition:
            w ≈ α₁s₁ + α₂s₂ + ... + αₖsₖ + ε

        and computes inter-sense angles, coefficients, dimensional
        territories, and interference patterns.

        Args:
            word: Target polysemous word
            senses: Pre-extracted sense vectors {label: vector}.
                    If None, senses are extracted via `mode`.
            mode: How to obtain senses if not provided:
                  'induce' (default): weakly supervised via anchors
                  'discover': unsupervised spectral clustering
                  'discover_auto': unsupervised + auto k
            force: Force re-extraction even if cached
            **kwargs: Passed to the sense extraction method

        Returns:
            SenseDecomposition with full geometric analysis

        Raises:
            ImportError: If geometry module is not available
            ValueError: If word not in vocabulary or < 2 senses found

        Example:
            >>> se = SenseExplorer.from_glove("glove.6B.100d.txt")
            >>> decomp = se.localize_senses("bank")
            >>> print(f"R² = {decomp.variance_explained_total:.3f}")
            >>> for s1, s2, angle in decomp.angle_pairs:
            ...     print(f"  ∠({s1}, {s2}) = {angle:.1f}°")
        """
        if not GEOMETRY_AVAILABLE:
            raise ImportError(
                "Geometry module not available. Ensure geometry.py is in "
                "the sense_explorer package directory."
            )

        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")

        # Obtain sense vectors
        if senses is None:
            if mode == 'induce':
                senses = self.induce_senses(word, force=force, **kwargs)
            elif mode == 'discover':
                senses = self.discover_senses(word, force=force, **kwargs)
            elif mode == 'discover_auto':
                senses = self.discover_senses_auto(word, force=force, **kwargs)
            else:
                raise ValueError(
                    f"Unknown mode '{mode}'. "
                    "Use 'induce', 'discover', or 'discover_auto'."
                )

        if len(senses) < 2:
            raise ValueError(
                f"Need at least 2 senses for geometry analysis, "
                f"got {len(senses)} for '{word}'."
            )

        decomp = _decompose_geometry(word, self.embeddings[word], senses)

        if self.verbose:
            _print_geometry_report(decomp)

        return decomp

    def analyze_geometry(
        self,
        words: List[str],
        mode: str = 'induce',
        save_dir: str = None,
        verbose: bool = None,
        **kwargs
    ) -> List['SenseDecomposition']:
        """
        Run sense geometry analysis across multiple words.

        For each word, extracts senses and decomposes the word vector,
        then prints cross-word statistical summaries and optionally
        saves visualizations.

        Args:
            words: List of polysemous words to analyze
            mode: Sense extraction mode ('induce', 'discover', 'discover_auto')
            save_dir: If provided, save dashboards and summary plots here
            verbose: Override instance verbose setting
            **kwargs: Passed to localize_senses()

        Returns:
            List of SenseDecomposition objects

        Example:
            >>> results = se.analyze_geometry(
            ...     ["bank", "cell", "run"],
            ...     save_dir="geometry_output"
            ... )
            >>> # Access cross-word angle statistics
            >>> from sense_explorer.geometry import collect_all_angles
            >>> angles = collect_all_angles(results)
        """
        if not GEOMETRY_AVAILABLE:
            raise ImportError(
                "Geometry module not available. Ensure geometry.py is in "
                "the sense_explorer package directory."
            )

        v = verbose if verbose is not None else self.verbose
        old_verbose = self.verbose
        self.verbose = False  # Suppress per-word chatter during batch

        decompositions = []
        for word in words:
            if word not in self.vocab:
                if v:
                    print(f"  WARNING: '{word}' not in vocabulary, skipping")
                continue
            try:
                decomp = self.localize_senses(word, mode=mode, **kwargs)
                decompositions.append(decomp)
            except ValueError as e:
                if v:
                    print(f"  WARNING: {e}")

        self.verbose = old_verbose

        if not decompositions:
            if v:
                print("No successful decompositions.")
            return []

        # Print cross-word summary
        if v:
            _print_geometry_summary(decompositions)

        # Save visualizations if requested
        if save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)

            for decomp in decompositions:
                path = os.path.join(save_dir, f'dashboard_{decomp.word}.png')
                plot_word_dashboard(decomp, path)
                if v:
                    print(f"  Saved: {path}")

            path = os.path.join(save_dir, 'cross_word_comparison.png')
            plot_cross_word_comparison(decompositions, path)
            if v:
                print(f"  Saved: {path}")

            path = os.path.join(save_dir, 'angle_summary.png')
            plot_angle_summary(decompositions, path)
            if v:
                print(f"  Saved: {path}")

        return decompositions

    def induce_senses_stable(

        self,
        word: str,
        n_senses: int = None,
        anchors: Dict[str, List[str]] = None,
        noise_levels: List[float] = None,
        n_trials: int = 3
    ) -> Dict:
        """
        Induce senses using stability-based method.
        
        This method runs sense induction at multiple noise levels and
        finds the STABLE sense count - the one that persists across
        different granularities. This is more robust than single-noise
        discovery.
        
        Args:
            word: Target word
            n_senses: Max number of senses to consider
            anchors: Optional dict of {sense_name: [anchor_words]}
            noise_levels: List of noise levels to test (default: 10%-70%)
            n_trials: Number of trials per noise level
        
        Returns:
            Dict with:
                - 'senses': sense embeddings at optimal noise
                - 'stable_k': stable sense count
                - 'optimal_noise': best noise level
                - 'confidence': stability confidence
                - 'stable_range': (start, end) noise range
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        if noise_levels is None:
            noise_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        
        n_senses = n_senses or 6  # Allow more senses for exploration
        
        # Get or induce anchors
        if anchors is None:
            anchors = self._extract_anchors_hybrid(word, n_senses)
        
        sense_centroids = self._compute_sense_centroids(anchors)
        
        if len(sense_centroids) < 2:
            return {
                'senses': {'default': self._embeddings_norm[word]},
                'stable_k': 1,
                'optimal_noise': self.noise_level,
                'confidence': 1.0,
                'stable_range': None
            }
        
        # Collect sense counts at each noise level
        sense_counts = []
        results_by_noise = {}
        
        for noise in noise_levels:
            trial_counts = []
            trial_results = []
            
            for _ in range(n_trials):
                sense_embs = self._simulated_repair(word, sense_centroids, noise, sense_loyal=True)
                k = len(sense_embs)
                trial_counts.append(k)
                trial_results.append(sense_embs)
            
            # Use mode (most common count)
            mode_count = Counter(trial_counts).most_common(1)[0][0]
            sense_counts.append((noise, mode_count))
            
            # Store result with mode count
            for embs in trial_results:
                if len(embs) == mode_count:
                    results_by_noise[noise] = embs
                    break
        
        # Find longest stable run
        best_run = None
        best_length = 0
        current_k = None
        current_start = None
        
        for noise, k in sense_counts:
            if k != current_k:
                if current_k is not None and current_start is not None:
                    length = noise - current_start
                    if length > best_length:
                        best_length = length
                        best_run = (current_start, noise, current_k)
                current_k = k
                current_start = noise
        
        # Check final run
        if current_k is not None and current_start is not None:
            length = sense_counts[-1][0] - current_start + (noise_levels[1] - noise_levels[0])
            if length > best_length:
                best_length = length
                best_run = (current_start, sense_counts[-1][0], current_k)
        
        if best_run is None:
            # No stable run found, use most common count
            all_counts = [k for _, k in sense_counts]
            stable_k = Counter(all_counts).most_common(1)[0][0]
            stable_range = None
            confidence = 0.0
            optimal_noise = self.noise_level
        else:
            stable_k = best_run[2]
            stable_range = (best_run[0], best_run[1])
            confidence = best_length / (noise_levels[-1] - noise_levels[0])
            optimal_noise = (stable_range[0] + stable_range[1]) / 2
        
        # Get senses at optimal noise
        if optimal_noise in results_by_noise:
            final_senses = results_by_noise[optimal_noise]
        else:
            final_senses = self._simulated_repair(word, sense_centroids, optimal_noise, sense_loyal=True)
        
        # Cache
        self._sense_cache[word] = final_senses
        self._stability_cache[word] = {
            'stable_k': stable_k,
            'optimal_noise': optimal_noise,
            'confidence': confidence,
            'stable_range': stable_range,
            'sense_counts': sense_counts
        }
        
        return {
            'senses': final_senses,
            'stable_k': stable_k,
            'optimal_noise': optimal_noise,
            'confidence': confidence,
            'stable_range': stable_range
        }
    
    def _discover_anchors_kmeans(self, word: str, n_senses: int) -> Dict[str, List[str]]:
        """
        UNSUPERVISED: Discover anchors via k-means clustering of nearest neighbors.
        
        This is used for true sense discovery mode (no external knowledge).
        Achieves ~56% accuracy.
        """
        target_emb = self._embeddings_norm[word]
        target_dim = len(target_emb)
        
        # Find nearest neighbors (excluding the word itself)
        k_neighbors = min(100, self.vocab_size - 1)
        
        similarities = []
        for w, emb in self._embeddings_norm.items():
            if w == word:
                continue
            # Handle dimension mismatch
            min_dim = min(target_dim, len(emb))
            sim = np.dot(target_emb[:min_dim], emb[:min_dim])
            similarities.append((w, sim, self.embeddings[w]))
        
        similarities.sort(key=lambda x: -x[1])
        neighbors = similarities[:k_neighbors]
        
        neighbor_words = [n[0] for n in neighbors]
        # Ensure all neighbor vectors have consistent dimensions
        neighbor_vecs = []
        for n in neighbors:
            vec = n[2]
            if len(vec) < target_dim:
                # Pad with zeros if shorter
                vec = np.concatenate([vec, np.zeros(target_dim - len(vec))])
            elif len(vec) > target_dim:
                # Truncate if longer
                vec = vec[:target_dim]
            neighbor_vecs.append(vec)
        neighbor_vecs = np.array(neighbor_vecs)
        
        # Normalize for clustering
        norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True) + 1e-10
        neighbor_vecs_norm = neighbor_vecs / norms
        
        # K-means clustering
        assignments, centroids = self._kmeans(neighbor_vecs_norm, n_senses)
        
        # Group neighbors by cluster
        anchors = {}
        for sense_idx in range(n_senses):
            mask = assignments == sense_idx
            cluster_words = [neighbor_words[i] for i in range(len(neighbor_words)) if mask[i]]
            
            if cluster_words:
                # Use top n_anchors words from each cluster
                anchors[f"sense_{sense_idx}"] = cluster_words[:self.n_anchors]
        
        return anchors
    
    def _discover_anchors_xmeans(
        self, 
        word: str, 
        max_senses: int = 6,
        min_senses: int = 2
    ) -> Tuple[Dict[str, List[str]], int]:
        """
        UNSUPERVISED + PARAMETER-FREE: Discover anchors via X-means clustering.
        
        X-means automatically determines the optimal number of clusters using
        BIC (Bayesian Information Criterion). This makes sense discovery
        truly parameter-free.
        
        Args:
            word: Target word
            max_senses: Maximum number of senses to consider
            min_senses: Minimum number of senses to consider
        
        Returns:
            Tuple of (anchors dict, optimal_k)
        """
        target_emb = self._embeddings_norm[word]
        target_dim = len(target_emb)
        
        # Find nearest neighbors
        k_neighbors = min(100, self.vocab_size - 1)
        
        similarities = []
        for w, emb in self._embeddings_norm.items():
            if w == word:
                continue
            # Handle dimension mismatch
            min_dim = min(target_dim, len(emb))
            sim = np.dot(target_emb[:min_dim], emb[:min_dim])
            similarities.append((w, sim, self.embeddings[w]))
        
        similarities.sort(key=lambda x: -x[1])
        neighbors = similarities[:k_neighbors]
        
        neighbor_words = [n[0] for n in neighbors]
        # Ensure all neighbor vectors have consistent dimensions
        neighbor_vecs = []
        for n in neighbors:
            vec = n[2]
            if len(vec) < target_dim:
                # Pad with zeros if shorter
                vec = np.concatenate([vec, np.zeros(target_dim - len(vec))])
            elif len(vec) > target_dim:
                # Truncate if longer
                vec = vec[:target_dim]
            neighbor_vecs.append(vec)
        neighbor_vecs = np.array(neighbor_vecs)
        
        # Normalize for clustering
        norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True) + 1e-10
        neighbor_vecs_norm = neighbor_vecs / norms
        
        # X-means: try different k values and select by BIC
        best_bic = float('-inf')
        best_k = min_senses
        best_assignments = None
        best_centroids = None
        
        for k in range(min_senses, max_senses + 1):
            assignments, centroids = self._kmeans(neighbor_vecs_norm, k, max_iter=50)
            bic = self._compute_bic(neighbor_vecs_norm, assignments, centroids, k)
            
            if bic > best_bic:
                best_bic = bic
                best_k = k
                best_assignments = assignments
                best_centroids = centroids
        
        # Group neighbors by cluster
        anchors = {}
        for sense_idx in range(best_k):
            mask = best_assignments == sense_idx
            cluster_words = [neighbor_words[i] for i in range(len(neighbor_words)) if mask[i]]
            
            if cluster_words:
                anchors[f"sense_{sense_idx}"] = cluster_words[:self.n_anchors]
        
        return anchors, best_k
    
    def _compute_bic(
        self,
        data: np.ndarray,
        assignments: np.ndarray,
        centroids: np.ndarray,
        k: int
    ) -> float:
        """
        Compute Bayesian Information Criterion (BIC) for clustering.
        
        BIC = log-likelihood - (penalty for model complexity)
        Higher BIC = better model (balances fit vs complexity)
        
        Args:
            data: Data points (n_samples, n_features)
            assignments: Cluster assignments
            centroids: Cluster centroids
            k: Number of clusters
        
        Returns:
            BIC score (higher is better)
        """
        n_samples, n_features = data.shape
        
        # Compute within-cluster variance
        total_variance = 0.0
        cluster_sizes = []
        
        for cluster_idx in range(k):
            mask = assignments == cluster_idx
            cluster_size = np.sum(mask)
            cluster_sizes.append(cluster_size)
            
            if cluster_size > 1:
                cluster_points = data[mask]
                centroid = centroids[cluster_idx]
                # Sum of squared distances to centroid
                distances = np.sum((cluster_points - centroid) ** 2)
                total_variance += distances
        
        # Avoid division by zero
        if n_samples <= k:
            return float('-inf')
        
        # Estimate variance
        variance = total_variance / (n_samples - k) if (n_samples - k) > 0 else 1e-10
        variance = max(variance, 1e-10)  # Avoid log(0)
        
        # Log-likelihood (assuming Gaussian clusters)
        log_likelihood = 0.0
        for cluster_idx in range(k):
            n_i = cluster_sizes[cluster_idx]
            if n_i > 0:
                # Log-likelihood for this cluster
                log_likelihood += n_i * np.log(n_i / n_samples)  # Prior
                log_likelihood -= n_i * n_features / 2 * np.log(2 * np.pi * variance)  # Gaussian
                log_likelihood -= (n_i - 1) / 2  # Correction
        
        # Number of parameters: k centroids * n_features + k-1 mixing proportions + 1 variance
        n_params = k * n_features + k
        
        # BIC = log-likelihood - penalty
        bic = log_likelihood - n_params / 2 * np.log(n_samples)
        
        return bic
    
    def discover_senses_auto(
        self,
        word: str,
        max_senses: int = 5,
        min_senses: int = 2,
        noise_level: float = None,
        clustering_method: str = None,
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        PARAMETER-FREE sense discovery with automatic k selection.
        
        Automatically determines the optimal number of senses.
        
        With spectral clustering (default): uses eigengap heuristic (90% at 50d)
        With X-means: uses BIC criterion (64% at 50d)
        
        Args:
            word: Target word
            max_senses: Maximum senses to consider (default: 5)
            min_senses: Minimum senses to consider (default: 2)
            noise_level: Override default noise level
            clustering_method: Override default method ('spectral' or 'xmeans')
            force: Force rediscovery even if cached
        
        Returns:
            Dict mapping sense names to sense-specific embeddings
        
        Example:
            >>> se = SenseExplorer.from_glove("glove.txt")
            >>> senses = se.discover_senses_auto("bank")  # No n_senses needed!
            >>> print(f"Found {len(senses)} senses: {list(senses.keys())}")
            Found 2 senses: ['sense_0', 'sense_1']
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        cache_key = f"{word}_auto_discovered"
        if not force and cache_key in self._sense_cache:
            return self._sense_cache[cache_key]
        
        noise = noise_level if noise_level is not None else self.noise_level
        method = clustering_method or self.clustering_method
        
        # Select clustering method
        if method == 'spectral' and SPECTRAL_AVAILABLE:
            anchors, optimal_k = discover_anchors_spectral(
                word,
                self._embeddings_norm,
                self.embeddings,
                self.vocab,
                n_anchors=self.n_anchors,
                top_k=self.top_k,
                min_senses=min_senses,
                max_senses=max_senses
            )
            method_name = "SPECTRAL"
        else:
            # Fall back to X-means
            anchors, optimal_k = self._discover_anchors_xmeans(word, max_senses, min_senses)
            method_name = "X-MEANS"
        
        if self.verbose:
            print(f"  [{method_name}] Auto-discovered {optimal_k} senses for '{word}'")
        
        sense_centroids = self._compute_sense_centroids(anchors)
        
        if len(sense_centroids) < 2:
            emb = self._embeddings_norm[word]
            self._sense_cache[cache_key] = {'default': emb}
            return {'default': emb}
        
        sense_embs = self._simulated_repair(word, sense_centroids, noise, sense_loyal=False)
        
        self._sense_cache[cache_key] = sense_embs
        self._anchor_cache[cache_key] = anchors
        
        return sense_embs
    
    def _validate_anchors(
        self,
        word: str,
        anchors: Dict[str, List[str]],
        warn: bool = True
    ) -> Dict[str, dict]:
        """
        Validate anchor quality before self-repair.
        
        Performs three checks critical for sense induction correctness:
          1. Intra-sense coherence: anchors within a sense should agree
          2. Inter-sense separation: different senses should point apart
          3. Target relevance: anchors should relate to the target word
        
        Without validation, bad anchors silently produce separated but
        semantically wrong senses (our experiments show 100% separation
        but only 6.4% true alignment with random anchors).
        
        Args:
            word: Target word being disambiguated
            anchors: Dict of {sense_name: [anchor_words]}
            warn: If True, emit warnings for low-quality anchors
        
        Returns:
            Dict of {sense_name: {
                'coherence': float,   # Mean pairwise cosine within sense (0-1)
                'separation': float,  # Min cosine distance to other sense centroids
                'relevance': float,   # Mean cosine similarity to target word
                'quality': str,       # 'good', 'fair', or 'poor'
                'n_valid': int        # Number of anchors found in vocabulary
            }}
        """
        if word not in self.vocab:
            return {}
        
        target_vec = self._embeddings_norm[word]
        
        # Compute centroids and anchor vectors per sense
        sense_info = {}
        sense_centroids = {}
        
        for sense_name, anchor_words in anchors.items():
            valid_words = [w for w in anchor_words if w in self.vocab]
            if not valid_words:
                sense_info[sense_name] = {
                    'coherence': 0.0, 'separation': 0.0, 'relevance': 0.0,
                    'quality': 'poor', 'n_valid': 0
                }
                continue
            
            vecs = np.array([self._embeddings_norm[w] for w in valid_words])
            centroid = vecs.mean(axis=0)
            centroid = centroid / (norm(centroid) + 1e-10)
            sense_centroids[sense_name] = centroid
            
            # 1. Intra-sense coherence: mean pairwise cosine
            if len(vecs) > 1:
                sim_matrix = vecs @ vecs.T
                n = len(vecs)
                # Extract upper triangle (exclude diagonal)
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                coherence = float(sim_matrix[mask].mean())
            else:
                coherence = 1.0  # Single anchor is trivially coherent
            
            # 3. Target relevance: mean cosine to target word
            relevance = float((vecs @ target_vec).mean())
            
            sense_info[sense_name] = {
                'coherence': coherence,
                'separation': 0.0,  # Computed after all centroids are built
                'relevance': relevance,
                'quality': 'pending',
                'n_valid': len(valid_words)
            }
        
        # 2. Inter-sense separation: min cosine distance between centroid pairs
        sense_names = [s for s in sense_centroids]
        for i, s1 in enumerate(sense_names):
            min_sep = 1.0
            for j, s2 in enumerate(sense_names):
                if i != j:
                    cos_sim = float(sense_centroids[s1] @ sense_centroids[s2])
                    distance = 1.0 - cos_sim
                    min_sep = min(min_sep, distance)
            sense_info[s1]['separation'] = min_sep if len(sense_names) > 1 else 0.0
        
        # Assign quality labels
        for sense_name, info in sense_info.items():
            if info['n_valid'] == 0:
                info['quality'] = 'poor'
            elif info['coherence'] >= 0.3 and info['relevance'] >= 0.2:
                if len(sense_names) <= 1 or info['separation'] >= 0.1:
                    info['quality'] = 'good'
                else:
                    info['quality'] = 'fair'
            elif info['coherence'] >= 0.15 or info['relevance'] >= 0.1:
                info['quality'] = 'fair'
            else:
                info['quality'] = 'poor'
        
        # Emit warnings
        if warn:
            poor_senses = [s for s, info in sense_info.items() if info['quality'] == 'poor']
            if poor_senses:
                warnings.warn(
                    f"Low-quality anchors for '{word}' senses: {poor_senses}. "
                    f"Anchor quality determines sense correctness—consider providing "
                    f"better anchor words. (See _validate_anchors() output for details.)"
                )
            
            # Check for overlapping senses
            for i, s1 in enumerate(sense_names):
                for j, s2 in enumerate(sense_names):
                    if i < j and sense_info[s1]['separation'] < 0.05:
                        warnings.warn(
                            f"Senses '{s1}' and '{s2}' have nearly identical anchors "
                            f"(separation={sense_info[s1]['separation']:.3f}). "
                            f"They may collapse to the same sense."
                        )
        
        return sense_info
    
    def _compute_sense_centroids(self, anchors: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """Compute normalized centroid for each sense from anchor words."""
        centroids = {}
        
        for sense_name, anchor_words in anchors.items():
            anchor_vecs = [self.embeddings[w] for w in anchor_words if w in self.vocab]
            if anchor_vecs:
                centroid = np.mean(anchor_vecs, axis=0)
                centroid = centroid / (norm(centroid) + 1e-10)
                centroids[sense_name] = centroid
        
        return centroids
    
    def _simulated_repair(
        self,
        word: str,
        sense_centroids: Dict[str, np.ndarray],
        noise_level: float,
        sense_loyal: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run simulated self-repair to discover sense-specific embeddings.
        
        This is the core algorithm (attractor-following):
          1. Create seeded noisy copies (each copy is seeded toward a specific sense)
          2. Iteratively pull copies toward sense centroids (attractors)
          3. Average copies by sense assignment
        
        The algorithm converges because anchor centroids define deterministic
        attractors in embedding space. Seeding ensures copies start in the
        correct attractor basin, and iterative pull follows the gradient to
        the attractor. N (copies per sense) only reduces variance around the
        attractor—even N=1 would converge to approximately the right location.
        
        CRITICAL: Success depends on anchor quality, not N. Good anchors yield
        correct attractors; random anchors yield separated but wrong senses.
        
        Args:
            word: Target word
            sense_centroids: Dict of sense_name -> centroid vector
            noise_level: Amount of noise to add
            sense_loyal: If True, copies stay with seeded sense (for induction).
                        If False, copies can switch to nearest sense (for discovery).
        
        Note on sense_loyal:
          - True (default): For INDUCTION - ensures all specified senses are found
          - False: For DISCOVERY - allows natural attractor dynamics
        """
        embedding = self.embeddings[word]
        senses = list(sense_centroids.keys())
        n_senses = len(senses)
        copies_per_sense = self.n_copies // n_senses
        total_copies = copies_per_sense * n_senses
        
        # Create seeded noisy copies (vectorized)
        # Tile embedding for all copies at once
        copies = np.tile(embedding, (total_copies, 1))
        copy_sense_ids = np.repeat(np.arange(n_senses), copies_per_sense)
        centroid_matrix = np.array([sense_centroids[s] for s in senses])
        
        # Vectorized noise: generate mask and noise for all copies at once
        # Each copy perturbs 50-80% of dimensions
        n_perturb_min = int(self.dim * 0.5)
        n_perturb_max = int(self.dim * 0.8)
        noise_matrix = np.random.randn(total_copies, self.dim) * np.abs(embedding) * noise_level
        # Create random binary mask: each copy gets ~65% of dims perturbed
        mask_probs = np.random.uniform(0, 1, (total_copies, self.dim))
        frac = np.random.uniform(0.5, 0.8, (total_copies, 1))  # per-copy fraction
        noise_mask = (mask_probs < frac).astype(np.float32)
        copies += noise_matrix * noise_mask
        
        # Vectorized seeding toward sense centroids
        copy_norms = np.linalg.norm(copies, axis=1, keepdims=True) + 1e-10
        copies_normed = copies / copy_norms
        targets = centroid_matrix[copy_sense_ids]  # (total_copies, dim)
        directions = targets - copies_normed
        copies += self.seed_strength * directions * copy_norms
        
        # Self-organization iterations (fully vectorized)
        current_pull = self.anchor_pull
        
        for _ in range(self.n_iterations):
            norms = np.linalg.norm(copies, axis=1, keepdims=True) + 1e-10
            copies_norm = copies / norms
            
            if sense_loyal:
                # SENSE-LOYAL: Pull toward SEEDED sense (vectorized)
                targets = centroid_matrix[copy_sense_ids]
                directions = targets - copies_norm
                copies += current_pull * directions * norms
            else:
                # COMPETITIVE: Pull toward NEAREST sense (vectorized)
                similarities = copies_norm @ centroid_matrix.T
                assignments = np.argmax(similarities, axis=1)
                targets = centroid_matrix[assignments]
                directions = targets - copies_norm
                copies += current_pull * directions * norms
            
            current_pull *= 0.95  # Decay
        
        # Final sense assignment
        norms = np.linalg.norm(copies, axis=1, keepdims=True) + 1e-10
        copies_norm = copies / norms
        
        if sense_loyal:
            # Use seeded sense assignment
            final_assignments = copy_sense_ids
            min_copies = 1  # Keep all senses that have at least 1 copy
        else:
            # Use nearest sense assignment
            similarities = copies_norm @ centroid_matrix.T
            final_assignments = np.argmax(similarities, axis=1)
            min_copies = self.n_copies * 0.05  # At least 5% threshold
        
        # Average copies by sense
        sense_embeddings = {}
        
        for sense_idx, sense in enumerate(senses):
            mask = final_assignments == sense_idx
            if mask.sum() >= min_copies:
                sense_emb = copies[mask].mean(axis=0)
                sense_emb = sense_emb / (norm(sense_emb) + 1e-10)
                sense_embeddings[sense] = sense_emb
        
        return sense_embeddings
    
    def _kmeans(self, vectors: np.ndarray, k: int, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """K-means clustering with k-means++ initialization."""
        n, d = vectors.shape
        
        if n < k:
            k = n
        
        # K-means++ initialization
        centroids = np.zeros((k, d))
        centroids[0] = vectors[np.random.randint(n)]
        
        for i in range(1, k):
            dists = np.min([np.sum((vectors - centroids[j])**2, axis=1) 
                            for j in range(i)], axis=0)
            dists = np.maximum(dists, 1e-10)
            probs = dists / dists.sum()
            probs = probs / probs.sum()
            centroids[i] = vectors[np.random.choice(n, p=probs)]
        
        assignments = np.zeros(n, dtype=int)
        
        for _ in range(max_iter):
            distances = np.array([np.sum((vectors - centroids[j])**2, axis=1) for j in range(k)]).T
            new_assignments = np.argmin(distances, axis=1)
            
            if np.all(new_assignments == assignments):
                break
            assignments = new_assignments
            
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    centroids[j] = vectors[mask].mean(axis=0)
        
        return assignments, centroids
    
    # =========================================================================
    # Similarity Methods
    # =========================================================================
    
    def similarity(
        self,
        word1: str,
        word2: str,
        sense_aware: bool = True,
        context: List[str] = None
    ) -> float:
        """
        Compute similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            sense_aware: If True, use sense-aware similarity
            context: Optional context words for disambiguation
        
        Returns:
            Cosine similarity score
        """
        if word1 not in self.vocab or word2 not in self.vocab:
            return 0.0
        
        if not sense_aware:
            return self._cosine(word1, word2)
        
        if context:
            return self.context_similarity(word1, word2, context)[0]
        else:
            return self.max_sense_similarity(word1, word2)[0]
    
    def _cosine(self, word1: str, word2: str) -> float:
        """Standard cosine similarity."""
        return float(np.dot(self._embeddings_norm[word1], self._embeddings_norm[word2]))
    
    def max_sense_similarity(self, word1: str, word2: str) -> Tuple[float, str]:
        """
        Max-sense similarity: max_i sim(sense_i(w1), w2)
        
        Returns:
            Tuple of (similarity, selected_sense_name)
        """
        senses1 = self.induce_senses(word1)
        e2 = self._embeddings_norm[word2]
        
        best_sim = -1.0
        best_sense = 'default'
        
        for sense_name, sense_emb in senses1.items():
            sim = np.dot(sense_emb, e2)
            if sim > best_sim:
                best_sim = sim
                best_sense = sense_name
        
        return float(best_sim), best_sense
    
    def best_match_similarity(self, word1: str, word2: str) -> Tuple[float, str, str]:
        """
        Best-match similarity for two potentially polysemous words.
        
        Returns:
            Tuple of (similarity, word1_sense, word2_sense)
        """
        senses1 = self.induce_senses(word1)
        senses2 = self.induce_senses(word2)
        
        best_sim = -1.0
        best_s1 = 'default'
        best_s2 = 'default'
        
        for s1_name, s1_emb in senses1.items():
            for s2_name, s2_emb in senses2.items():
                sim = np.dot(s1_emb, s2_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_s1 = s1_name
                    best_s2 = s2_name
        
        return float(best_sim), best_s1, best_s2
    
    def context_similarity(
        self,
        word1: str,
        word2: str,
        context: List[str]
    ) -> Tuple[float, str, str]:
        """
        Context-disambiguated similarity.
        
        Args:
            word1: First word
            word2: Second word
            context: Context words for disambiguation
        
        Returns:
            Tuple of (similarity, word1_sense, word2_sense)
        """
        senses1 = self.induce_senses(word1)
        senses2 = self.induce_senses(word2)
        
        # Compute context vector
        context_vecs = [self._embeddings_norm[c] for c in context if c in self.vocab]
        if not context_vecs:
            return self.best_match_similarity(word1, word2)
        
        context_mean = np.mean(context_vecs, axis=0)
        context_mean = context_mean / (norm(context_mean) + 1e-10)
        
        # Select senses aligned with context
        best_s1 = max(senses1.items(), key=lambda x: np.dot(x[1], context_mean))[0]
        best_s2 = max(senses2.items(), key=lambda x: np.dot(x[1], context_mean))[0]
        
        sim = np.dot(senses1[best_s1], senses2[best_s2])
        
        return float(sim), best_s1, best_s2
    
    # =========================================================================
    # Analogy
    # =========================================================================
    
    def analogy(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 10,
        sense_aware: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Solve analogy: a:b :: c:?
        
        Args:
            a, b, c: Analogy words
            top_k: Number of results to return
            sense_aware: If True, use sense-aware analogy
        
        Returns:
            List of (word, score) tuples
        """
        if a not in self.vocab or b not in self.vocab or c not in self.vocab:
            return []
        
        exclude = {a, b, c}
        
        if sense_aware:
            senses_a = self.induce_senses(a)
            senses_c = self.induce_senses(c)
            e_b = self._embeddings_norm[b]
            
            # Find which sense of 'a' relates to 'b'
            best_a_sense = max(senses_a.items(), key=lambda x: np.dot(x[1], e_b))[0]
            e_a = senses_a[best_a_sense]
            
            # Use matching sense of 'c' if available
            if best_a_sense in senses_c:
                e_c = senses_c[best_a_sense]
            else:
                e_c = max(senses_c.items(), key=lambda x: np.dot(x[1], e_a))[1]
        else:
            e_a = self._embeddings_norm[a]
            e_b = self._embeddings_norm[b]
            e_c = self._embeddings_norm[c]
        
        # Compute analogy vector
        analogy_vec = e_b - e_a + e_c
        analogy_vec = analogy_vec / (norm(analogy_vec) + 1e-10)
        
        # Find nearest neighbors
        results = []
        for word, emb in self._embeddings_norm.items():
            if word in exclude:
                continue
            score = np.dot(analogy_vec, emb)
            results.append((word, float(score)))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_senses(self, word: str) -> List[str]:
        """Get list of discovered sense names for a word."""
        if word in self._sense_cache:
            return list(self._sense_cache[word].keys())
        return []
    
    def get_sense_embedding(self, word: str, sense: str) -> Optional[np.ndarray]:
        """Get embedding for a specific sense of a word."""
        if word in self._sense_cache and sense in self._sense_cache[word]:
            return self._sense_cache[word][sense].copy()
        return None
    
    def get_anchors(self, word: str) -> Optional[Dict[str, List[str]]]:
        """Get anchor words used for each sense."""
        return self._anchor_cache.get(word)
    
    def get_stability_info(self, word: str) -> Optional[Dict]:
        """Get stability analysis results for a word."""
        return self._stability_cache.get(word)
    
    def clear_cache(self):
        """Clear all cached sense embeddings."""
        self._sense_cache.clear()
        self._anchor_cache.clear()
        self._stability_cache.clear()
    
    def set_anchors(self, word: str, anchors: Dict[str, List[str]]):
        """
        Manually set anchors for a word (useful for known polysemous words).
        
        Example:
            sr.set_anchors('bank', {
                'financial': ['money', 'account', 'loan'],
                'river': ['river', 'shore', 'water']
            })
        """
        # Clear existing cache for this word
        if word in self._sense_cache:
            del self._sense_cache[word]
        if word in self._stability_cache:
            del self._stability_cache[word]
        
        # Induce senses with these anchors
        self.induce_senses(word, anchors=anchors, force=True)
    
    def set_noise_level(self, noise_level: float):
        """
        Set the noise level (granularity control).
        
        Guidelines:
            - 0.1-0.2: Fine-grained (may over-split)
            - 0.3-0.5: Standard sense-level (recommended)
            - 0.6-0.8: Coarse-grained
        
        Higher noise = coarser sense distinctions
        """
        self.noise_level = noise_level
        # Clear cache since granularity changed
        self.clear_cache()
    
    # =========================================================================
    # Polarity Methods (97% accuracy)
    # =========================================================================
    
    def get_polarity(
        self,
        word: str,
        positive_seeds: List[str] = None,
        negative_seeds: List[str] = None
    ) -> Dict:
        """
        Get polarity classification for a word.
        
        This is a SUPERVISED method: requires seed words to define
        the positive/negative poles. Achieves 97% accuracy.
        
        Args:
            word: Target word
            positive_seeds: Words defining positive pole (default: sentiment words)
            negative_seeds: Words defining negative pole
        
        Returns:
            Dict with 'polarity', 'score', 'confidence'
        
        Example:
            >>> se.get_polarity("excellent")
            {'polarity': 'positive', 'score': 0.82, 'confidence': 0.91}
        """
        # Lazy import to avoid circular dependency
        from .polarity import PolarityFinder, DEFAULT_POLARITY_SEEDS
        
        # Use cached polarity finder or create new one
        cache_key = (tuple(positive_seeds) if positive_seeds else None,
                     tuple(negative_seeds) if negative_seeds else None)
        
        if not hasattr(self, '_polarity_finders'):
            self._polarity_finders = {}
        
        if cache_key not in self._polarity_finders:
            pf = PolarityFinder(
                self.embeddings,
                positive_seeds=positive_seeds,
                negative_seeds=negative_seeds,
                verbose=False
            )
            self._polarity_finders[cache_key] = pf
        
        return self._polarity_finders[cache_key].get_polarity(word)
    
    def classify_polarity(
        self,
        words: List[str],
        positive_seeds: List[str] = None,
        negative_seeds: List[str] = None,
        threshold: float = 0.0
    ) -> Dict[str, List[str]]:
        """
        Classify multiple words by polarity.
        
        Args:
            words: Words to classify
            positive_seeds: Words defining positive pole
            negative_seeds: Words defining negative pole
            threshold: Score threshold (|score| < threshold = neutral)
        
        Returns:
            Dict with 'positive', 'negative', 'neutral' word lists
        
        Example:
            >>> se.classify_polarity(['good', 'bad', 'table'])
            {'positive': ['good'], 'negative': ['bad'], 'neutral': ['table']}
        """
        from .polarity import PolarityFinder
        
        pf = PolarityFinder(
            self.embeddings,
            positive_seeds=positive_seeds,
            negative_seeds=negative_seeds,
            verbose=False
        )
        return pf.classify_words(words, threshold=threshold)
    
    def get_polarity_finder(
        self,
        positive_seeds: List[str] = None,
        negative_seeds: List[str] = None,
        domain: str = None
    ) -> 'PolarityFinder':
        """
        Get a PolarityFinder instance for advanced polarity operations.
        
        Args:
            positive_seeds: Custom positive seeds
            negative_seeds: Custom negative seeds  
            domain: Predefined domain ('sentiment', 'quality', 'morality',
                   'health', 'size', 'temperature')
        
        Returns:
            PolarityFinder instance
        
        Example:
            >>> pf = se.get_polarity_finder(domain='quality')
            >>> pf.most_polar_words(top_k=10)
        """
        from .polarity import PolarityFinder, DOMAIN_POLARITY_SEEDS
        
        if domain and domain in DOMAIN_POLARITY_SEEDS:
            positive_seeds = DOMAIN_POLARITY_SEEDS[domain]['positive']
            negative_seeds = DOMAIN_POLARITY_SEEDS[domain]['negative']
        
        return PolarityFinder(
            self.embeddings,
            positive_seeds=positive_seeds,
            negative_seeds=negative_seeds,
            verbose=self.verbose
        )
    
    # =========================================================================
    # Embedding Merger Methods (NEW in v0.9.3)
    # =========================================================================
    
    def get_merger(
        self,
        other_explorers: Dict[str, 'SenseExplorer'] = None,
        **kwargs
    ) -> 'EmbeddingMerger':
        """
        Get an EmbeddingMerger configured with this explorer's embeddings.
        
        Args:
            other_explorers: Additional SenseExplorer instances to include.
                             Dict mapping names to explorers.
            **kwargs: Additional arguments for EmbeddingMerger
        
        Returns:
            Configured EmbeddingMerger
            
        Example:
            >>> se_wiki = SenseExplorer.from_file("wiki.txt")
            >>> se_twitter = SenseExplorer.from_file("twitter.txt")
            >>> 
            >>> # Get merger from wiki, adding twitter
            >>> merger = se_wiki.get_merger({"twitter": se_twitter})
            >>> result = merger.merge_senses("bank")
        """
        if not MERGER_AVAILABLE:
            raise ImportError(
                "Merger module not available. "
                "Ensure merger.py is in the sense_explorer package."
            )
        
        merger = EmbeddingMerger(verbose=self.verbose, **kwargs)
        merger.add_embedding("self", self.embeddings)
        
        if other_explorers:
            for name, se in other_explorers.items():
                merger.add_embedding(name, se.embeddings)
        
        return merger
    
    def merge_with(
        self,
        other: 'SenseExplorer',
        word: str,
        other_name: str = "other",
        self_name: str = "self",
        n_senses: int = None,
        use_ssr: bool = True,
        distance_threshold: float = 0.05
    ) -> 'MergerResult':
        """
        Merge senses of a word with another embedding.
        
        This is a convenience method for two-embedding merger.
        For more control or more embeddings, use get_merger().
        
        Args:
            other: Another SenseExplorer instance
            word: Word to merge
            other_name: Name for the other embedding (for labeling)
            self_name: Name for this embedding (for labeling)
            n_senses: Number of senses to extract (None = auto)
            use_ssr: If True, use SSR for sense extraction; else use k-means
            distance_threshold: Clustering threshold
            
        Returns:
            MergerResult with convergent and source-specific senses
            
        Example:
            >>> se_wiki = SenseExplorer.from_file("wiki.txt")
            >>> se_twitter = SenseExplorer.from_file("twitter.txt")
            >>> 
            >>> result = se_wiki.merge_with(se_twitter, "bank")
            >>> print(f"Convergent senses: {result.n_convergent}")
            >>> print(f"Source-specific: {result.n_source_specific}")
        """
        if not MERGER_AVAILABLE:
            raise ImportError(
                "Merger module not available. "
                "Ensure merger.py is in the sense_explorer package."
            )
        
        if use_ssr:
            # Use SSR-based extraction
            return merge_with_ssr(
                {self_name: self, other_name: other},
                word,
                n_senses=n_senses,
                distance_threshold=distance_threshold,
                verbose=self.verbose
            )
        else:
            # Use simple k-means extraction
            merger = EmbeddingMerger(verbose=self.verbose)
            merger.add_embedding(self_name, self.embeddings)
            merger.add_embedding(other_name, other.embeddings)
            return merger.merge_senses(
                word, 
                n_senses=n_senses or 3,
                distance_threshold=distance_threshold
            )
    
    def extract_sense_components(
        self,
        word: str,
        source_name: str = None,
        n_senses: int = None
    ) -> List['SenseComponent']:
        """
        Extract sense components in a format suitable for embedding merger.
        
        Uses SSR (induce_senses) for extraction, then wraps results
        in SenseComponent dataclass for use with EmbeddingMerger.
        
        Args:
            word: Target word
            source_name: Name for this embedding source
            n_senses: Number of senses (None = auto via eigengap)
            
        Returns:
            List of SenseComponent objects
            
        Example:
            >>> # Extract from multiple explorers
            >>> components = []
            >>> components.extend(se_wiki.extract_sense_components("bank", "wiki"))
            >>> components.extend(se_twitter.extract_sense_components("bank", "twitter"))
            >>> 
            >>> # Use with merger
            >>> merger = EmbeddingMerger()
            >>> merger.add_embedding("wiki", se_wiki.embeddings)
            >>> merger.add_embedding("twitter", se_twitter.embeddings)
            >>> result = merger.merge_senses("bank", sense_components=components)
        """
        if not MERGER_AVAILABLE:
            raise ImportError(
                "Merger module not available. "
                "Ensure merger.py is in the sense_explorer package."
            )
        
        if source_name is None:
            source_name = "source"
        
        # Use SSR to get sense vectors
        senses = self.induce_senses(word, n_senses=n_senses)
        
        components = []
        for sense_name, sense_vec in senses.items():
            # Compute neighbors
            neighbors = []
            for w in self.embeddings:
                if w != word:
                    sim = float(np.dot(sense_vec, self._embeddings_norm.get(w, self.embeddings[w])))
                    neighbors.append((w, sim))
            neighbors.sort(key=lambda x: -x[1])
            
            components.append(SenseComponent(
                word=word,
                sense_id=f"{source_name}_{sense_name}",
                vector=sense_vec,
                source=source_name,
                top_neighbors=neighbors[:50]
            ))
        
        return components
    
    # =========================================================================
    # Sense Distillation via IVA (v0.9.2)
    # =========================================================================
    
    def get_distiller(self) -> 'IVADistiller':
        """
        Get an IVA distiller using this explorer's embeddings.
        
        IVA (Iterative Vector Averaging) distills the shared semantic
        component from a set of words — complementing SSR's sense separation.
        
        Returns:
            IVADistiller instance
        
        Example:
            >>> distiller = se.get_distiller()
            >>> result = distiller.distill_constrained(['money', 'loan', 'account'])
            >>> print(result.exemplars)
        """
        if not DISTILLATION_AVAILABLE:
            raise ImportError(
                "Distillation module not available. "
                "Ensure distillation.py is in the sense_explorer package."
            )
        
        return IVADistiller(
            embeddings=self.embeddings,
            verbose=self.verbose
        )
    
    def distill_senses(
        self,
        word: str,
        mode: str = 'constrained',
        n_exemplars: int = 5,
        n_senses: int = None
    ) -> Dict[str, 'DistillationResult']:
        """
        Distill sense directions via IVA for a word's senses.
        
        This combines SSR sense separation with IVA distillation:
        1. SSR extracts sense vectors and their anchors
        2. IVA distills each anchor set to a purified direction
        
        Args:
            word: Target word
            mode: 'constrained' (recommended) or 'global'
            n_exemplars: Number of exemplars per sense
            n_senses: Number of senses (None = auto)
        
        Returns:
            Dict mapping sense names to DistillationResults
        
        Example:
            >>> results = se.distill_senses("bank")
            >>> for sense, result in results.items():
            ...     print(f"{sense}: coherence={result.coherence:.3f}")
            financial: coherence=0.456
            river: coherence=0.423
        """
        if not DISTILLATION_AVAILABLE:
            raise ImportError(
                "Distillation module not available. "
                "Ensure distillation.py is in the sense_explorer package."
            )
        
        # Get anchors for each sense
        anchors = self.get_anchors(word)
        
        if not anchors:
            # Induce senses first to populate anchors
            _ = self.induce_senses(word, n_senses=n_senses)
            anchors = self.get_anchors(word)
        
        if not anchors:
            if self.verbose:
                print(f"No anchors found for '{word}'")
            return {}
        
        # Distill each sense's anchors
        distiller = self.get_distiller()
        return distiller.distill_multiple(anchors, mode=mode, n_exemplars=n_exemplars)
    
    def measure_anchor_coherence(self, word: str) -> Dict[str, float]:
        """
        Measure coherence of anchor sets for a word's senses.
        
        Use this to validate anchor quality before distillation.
        Higher coherence → more meaningful distillation.
        
        Reference values:
            - Random words: ~0.15
            - Topically related: ~0.30-0.40
            - Sense-coherent: ~0.45-0.55
            - Near-synonyms: ~0.60-0.80
        
        Args:
            word: Target word
        
        Returns:
            Dict mapping sense names to coherence scores
        
        Example:
            >>> coherences = se.measure_anchor_coherence("bank")
            >>> print(coherences)
            {'financial': 0.456, 'river': 0.423}
        """
        if not DISTILLATION_AVAILABLE:
            raise ImportError("Distillation module not available.")
        
        anchors = self.get_anchors(word)
        
        if not anchors:
            _ = self.induce_senses(word)
            anchors = self.get_anchors(word)
        
        if not anchors:
            if self.verbose:
                print(f"No anchors found for '{word}'")
            return {}
        
        return {
            sense: measure_set_coherence(self.embeddings, words)
            for sense, words in anchors.items()
        }
    
    def distill_and_compare(
        self,
        word: str,
        mode: str = 'constrained'
    ) -> Dict[str, any]:
        """
        Distill senses and compare SSR vectors with IVA directions.
        
        This validates the relationship between SSR sense separation
        and IVA concept distillation.
        
        Args:
            word: Target word
            mode: 'constrained' or 'global'
        
        Returns:
            Dict with:
                - ssr_senses: SSR sense vectors
                - iva_results: IVA distillation results
                - ssr_iva_angles: Angles between SSR and IVA per sense
                - iva_inter_sense_angles: Angles between IVA directions
        
        Example:
            >>> stats = se.distill_and_compare("bank")
            >>> print(stats['ssr_iva_angles'])
            {'financial': 12.3, 'river': 8.7}
        """
        if not DISTILLATION_AVAILABLE:
            raise ImportError("Distillation module not available.")
        
        # Get SSR senses
        ssr_senses = self.induce_senses(word)
        
        # Get IVA distillations
        iva_results = self.distill_senses(word, mode=mode)
        
        # Compare SSR and IVA directions
        ssr_iva_angles = {}
        for sense_name in ssr_senses:
            if sense_name in iva_results:
                ssr_vec = ssr_senses[sense_name]
                iva_vec = iva_results[sense_name].direction
                
                # Normalize
                ssr_vec = ssr_vec / (np.linalg.norm(ssr_vec) + 1e-10)
                iva_vec = iva_vec / (np.linalg.norm(iva_vec) + 1e-10)
                
                # Compute angle
                cos_sim = np.clip(np.dot(ssr_vec, iva_vec), -1, 1)
                angle = np.degrees(np.arccos(cos_sim))
                ssr_iva_angles[sense_name] = angle
        
        # IVA inter-sense angles
        distiller = self.get_distiller()
        iva_inter_angles = distiller.compare_directions(iva_results)
        
        return {
            'ssr_senses': ssr_senses,
            'iva_results': iva_results,
            'ssr_iva_angles': ssr_iva_angles,
            'iva_inter_sense_angles': iva_inter_angles,
            'mean_ssr_iva_angle': np.mean(list(ssr_iva_angles.values())) if ssr_iva_angles else 0,
            'mean_iva_inter_angle': np.mean(list(iva_inter_angles.values())) if iva_inter_angles else 0,
        }
    
    def validate_sense_distillation(
        self,
        words: List[str],
        mode: str = 'constrained'
    ) -> Dict[str, any]:
        """
        Run batch validation of sense distillation across multiple words.
        
        Args:
            words: List of words to validate
            mode: 'constrained' or 'global'
        
        Returns:
            Dict with aggregate statistics
        """
        if not DISTILLATION_AVAILABLE:
            raise ImportError("Distillation module not available.")
        
        all_coherences = []
        all_ssr_iva_angles = []
        all_inter_sense_angles = []
        per_word = {}
        
        for word in words:
            if word not in self.vocab:
                continue
            
            try:
                stats = self.distill_and_compare(word, mode=mode)
                per_word[word] = stats
                
                # Collect coherences
                for result in stats['iva_results'].values():
                    all_coherences.append(result.coherence)
                
                # Collect angles
                all_ssr_iva_angles.extend(stats['ssr_iva_angles'].values())
                all_inter_sense_angles.extend(stats['iva_inter_sense_angles'].values())
            except Exception as e:
                if self.verbose:
                    print(f"  Skipping '{word}': {e}")
        
        return {
            'per_word': per_word,
            'mean_coherence': np.mean(all_coherences) if all_coherences else 0,
            'mean_ssr_iva_angle': np.mean(all_ssr_iva_angles) if all_ssr_iva_angles else 0,
            'mean_inter_sense_angle': np.mean(all_inter_sense_angles) if all_inter_sense_angles else 0,
            'n_words': len(per_word),
            'n_senses': len(all_coherences),
        }
    
    def __repr__(self) -> str:
        return f"SenseExplorer(vocab_size={self.vocab_size:,}, dim={self.dim}, cached_senses={len(self._sense_cache)})"


# =============================================================================
# Predefined Anchor Sets for Common Polysemous Words
# =============================================================================

COMMON_POLYSEMOUS = {
    'bank': {
        'financial': ['money', 'account', 'deposit', 'loan', 'finance', 'credit', 'savings', 'investment'],
        'river': ['river', 'shore', 'water', 'stream', 'edge', 'slope', 'embankment', 'side']
    },
    'bat': {
        'animal': ['bird', 'wing', 'fly', 'cave', 'mammal', 'nocturnal', 'vampire', 'creature'],
        'sports': ['ball', 'hit', 'swing', 'baseball', 'cricket', 'wooden', 'player', 'game']
    },
    'cell': {
        'biology': ['organism', 'membrane', 'nucleus', 'dna', 'protein', 'tissue', 'division', 'microscope'],
        'prison': ['jail', 'prisoner', 'locked', 'bars', 'inmate', 'detention', 'solitary', 'confined'],
        'phone': ['mobile', 'telephone', 'call', 'wireless', 'smartphone', 'signal', 'battery', 'tower']
    },
    'crane': {
        'bird': ['bird', 'wing', 'fly', 'nest', 'feather', 'migrate', 'wetland', 'heron'],
        'machine': ['construction', 'lift', 'heavy', 'tower', 'operator', 'load', 'steel', 'building']
    },
    'mouse': {
        'animal': ['rat', 'rodent', 'cat', 'trap', 'cheese', 'squirrel', 'pet', 'tail'],
        'computer': ['keyboard', 'click', 'cursor', 'screen', 'computer', 'button', 'pointer', 'device']
    },
    'plant': {
        'vegetation': ['tree', 'flower', 'leaf', 'grow', 'seed', 'garden', 'root', 'soil'],
        'factory': ['manufacturing', 'production', 'industrial', 'facility', 'equipment', 'machinery', 'worker', 'output']
    },
    'spring': {
        'season': ['summer', 'winter', 'autumn', 'march', 'april', 'bloom', 'warm', 'flowers'],
        'water': ['fountain', 'well', 'source', 'flow', 'natural', 'mineral', 'hot', 'fresh']
    },
    'bass': {
        'fish': ['fish', 'fishing', 'lake', 'catch', 'trout', 'salmon', 'angler', 'pond'],
        'music': ['guitar', 'music', 'band', 'drum', 'sound', 'instrument', 'player', 'rhythm']
    },
    'match': {
        'fire': ['lighter', 'flame', 'burn', 'ignite', 'stick', 'sulfur', 'strike', 'box'],
        'competition': ['game', 'tournament', 'opponent', 'winner', 'sports', 'contest', 'play', 'team']
    },
    'bow': {
        'weapon': ['arrow', 'archer', 'shoot', 'hunting', 'string', 'target', 'quiver', 'crossbow'],
        'gesture': ['curtsy', 'nod', 'greeting', 'respect', 'bend', 'head', 'polite', 'acknowledge']
    }
}


def load_common_polysemous(sr: SenseExplorer, words: List[str] = None):
    """
    Pre-load sense anchors for common polysemous words.
    
    Args:
        sr: SenseExplorer instance
        words: List of words to load (default: all common polysemous)
    """
    words = words or list(COMMON_POLYSEMOUS.keys())
    
    for word in words:
        if word in COMMON_POLYSEMOUS and word in sr.vocab:
            sr.set_anchors(word, COMMON_POLYSEMOUS[word])


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    """Command-line interface for SenseExplorer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SenseExplorer: Sense Discovery via Simulated Self-Repair',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Induce senses for a word
  python -m sense_explorer --glove glove.6B.100d.txt --word bank
  
  # Use stability-based discovery
  python -m sense_explorer --glove glove.6B.100d.txt --word bank --stable
  
  # Compute sense-aware similarity
  python -m sense_explorer --glove glove.6B.100d.txt --similarity bank river
  
  # Solve analogy
  python -m sense_explorer --glove glove.6B.100d.txt --analogy bank money crane
  
  # Control granularity with noise level
  python -m sense_explorer --glove glove.6B.100d.txt --word bank --noise 0.3
        """
    )
    
    parser.add_argument('--glove', type=str, help='Path to GloVe embeddings')
    parser.add_argument('--word2vec', type=str, help='Path to Word2Vec embeddings')
    parser.add_argument('--max-words', type=int, default=50000, help='Max words to load')
    parser.add_argument('--word', type=str, help='Word to discover senses for')
    parser.add_argument('--similarity', nargs=2, metavar=('W1', 'W2'), help='Compute similarity')
    parser.add_argument('--analogy', nargs=3, metavar=('A', 'B', 'C'), help='Solve analogy a:b::c:?')
    parser.add_argument('--n-senses', type=int, default=2, help='Number of senses to discover')
    parser.add_argument('--noise', type=float, default=0.5, help='Noise level (granularity control)')
    parser.add_argument('--stable', action='store_true', help='Use stability-based sense induction')
    parser.add_argument('--use-predefined', action='store_true', help='Use predefined anchors for common words')
    
    args = parser.parse_args()
    
    # Load embeddings
    if args.glove:
        sr = SenseExplorer.from_glove(args.glove, max_words=args.max_words, noise_level=args.noise)
    elif args.word2vec:
        sr = SenseExplorer.from_word2vec(args.word2vec, max_words=args.max_words, noise_level=args.noise)
    else:
        parser.error("Must specify --glove or --word2vec")
    
    # Load predefined anchors
    if args.use_predefined:
        load_common_polysemous(sr)
    
    # Induce senses
    if args.word:
        print(f"\nDiscovering senses for '{args.word}'...")
        
        if args.stable:
            result = sr.induce_senses_stable(args.word, n_senses=args.n_senses)
            print(f"Stable sense count: {result['stable_k']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Optimal noise: {result['optimal_noise']:.1%}")
            if result['stable_range']:
                print(f"Stable range: {result['stable_range'][0]:.0%}-{result['stable_range'][1]:.0%}")
            senses = result['senses']
        else:
            senses = sr.induce_senses(args.word, n_senses=args.n_senses)
        
        print(f"Found {len(senses)} senses: {list(senses.keys())}")
        
        anchors = sr.get_anchors(args.word)
        if anchors:
            print("\nAnchors used:")
            for sense, words in anchors.items():
                print(f"  {sense}: {', '.join(words[:5])}")
    
    # Compute similarity
    if args.similarity:
        w1, w2 = args.similarity
        
        std_sim = sr.similarity(w1, w2, sense_aware=False)
        max_sim, sense = sr.max_sense_similarity(w1, w2)
        
        print(f"\nSimilarity between '{w1}' and '{w2}':")
        print(f"  Standard:    {std_sim:.4f}")
        print(f"  Sense-aware: {max_sim:.4f} (using {sense} sense)")
        print(f"  Improvement: {max_sim - std_sim:+.4f}")
    
    # Solve analogy
    if args.analogy:
        a, b, c = args.analogy
        
        print(f"\nAnalogy: {a} : {b} :: {c} : ?")
        
        results = sr.analogy(a, b, c, top_k=5, sense_aware=True)
        print("Top answers (sense-aware):")
        for word, score in results:
            print(f"  {word}: {score:.4f}")


if __name__ == "__main__":
    main()
