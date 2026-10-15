"""
SenseExplorer: From Sense Discovery to Sense Induction via Simulated Self-Repair
=================================================================================

A lightweight, training-free framework for exploring word sense structure
in static embeddings using biologically-inspired self-repair.

Package Structure:
    sense_explorer (this package)
        ├── Basic: Sense separation from single embeddings
        │   ├── core.py          - SenseExplorer main class
        │   ├── spectral.py      - Spectral clustering
        │   ├── geometry.py      - Sense geometry analysis
        │   ├── polarity.py      - Polarity classification
        │   └── distillation.py  - IVA sense distillation
        │
        ├── merger/              - Advanced: Cross-embedding merger
        │   ├── embedding_merger.py        - Main merger
        │   └── staged_embedding_merger.py - Memory-efficient staged merger
        │
        └── Utilities: CLI export tools (installed as console scripts)
            ├── extract_senses_to_json.py   - SSR sense vectors -> JSON
            └── extract_baseline_to_json.py - Bare (no-SSR) vectors -> JSON, control condition

Key insight: The self-repair algorithm is attractor-following, not 
space-sampling. Anchor centroids define deterministic attractors—success
depends on anchor quality, not the number of noisy copies (N). Even N=3
achieves perfect separation when anchors are good.

Six capabilities on the supervision continuum:
  - discover_senses_auto():      Unsupervised - spectral clustering (90% at 50d)
  - discover_senses():           Semi-supervised - user specifies k
  - separate_senses_wordnet():   WordNet-guided - lexicographic structure + geometry
  - induce_senses():             Weakly supervised - anchor-guided (88% accuracy)
  - merge_with():                Cross-embedding - combine sense inventories (Advanced)
  - find_polarity():             Supervised - polarity classification (97% accuracy)

Basic Usage:
    >>> from sense_explorer import SenseExplorer
    >>> se = SenseExplorer.from_glove("glove.6B.300d.txt")
    
    # Unsupervised discovery (spectral, 90% at 50d)
    >>> senses = se.discover_senses_auto("bank")
    
    # WordNet-guided separation
    >>> senses = se.separate_senses_wordnet("bank")
    >>> print(senses.keys())  # Synset names as keys
    
    # Knowledge-guided induction (88% accuracy)
    >>> senses = se.induce_senses("bank")
    
    # Cross-embedding merger (Advanced - via subpackage)
    >>> from sense_explorer.merger import EmbeddingMerger
    >>> se_twitter = SenseExplorer.from_file("glove.twitter.100d.txt")
    >>> result = se.merge_with(se_twitter, "bank")
    >>> print(f"Convergent: {result.n_convergent}")
    
    # Polarity classification (97% accuracy)
    >>> polarity = se.get_polarity("excellent")

    # Sense geometry analysis (molecular bond analogy)
    >>> decomp = se.localize_senses("bank")
    >>> print(decomp.angle_pairs)  # Inter-sense angles

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
Version: 0.9.5
"""

from .core import (
    SenseExplorer,
    COMMON_POLYSEMOUS,
    load_common_polysemous
)

from .anchor_extractor import (
    HybridAnchorExtractor,
    extract_anchors,
    get_manual_anchors,
    get_frame_anchors,
    list_supported_words,
    MANUAL_ANCHORS,
    FRAME_ANCHORS
)

from .polarity import (
    PolarityFinder,
    DEFAULT_POLARITY_SEEDS,
    DOMAIN_POLARITY_SEEDS,
    classify_polarity
)

from .spectral import (
    spectral_clustering,
    discover_anchors_spectral,
    discover_anchors_spectral_fixed_k,
    find_k_by_eigengap
)

from .geometry import (
    SenseDecomposition,
    decompose,
    print_report,
    print_cross_word_summary,
    collect_all_angles,
    plot_word_dashboard,
    plot_molecular_diagram,
    plot_cross_word_comparison,
    plot_angle_summary,
)

# Distillation module (v0.9.2)
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

__version__ = "0.9.5"
__author__ = "Kow Kuroda & Claude"

__all__ = [
    # Core
    'SenseExplorer',
    # Anchors
    'HybridAnchorExtractor',
    'extract_anchors',
    'get_manual_anchors',
    'get_frame_anchors',
    'list_supported_words',
    'load_common_polysemous',
    'COMMON_POLYSEMOUS',
    'MANUAL_ANCHORS',
    'FRAME_ANCHORS',
    # Polarity
    'PolarityFinder',
    'DEFAULT_POLARITY_SEEDS',
    'DOMAIN_POLARITY_SEEDS',
    'classify_polarity',
    # Spectral
    'spectral_clustering',
    'discover_anchors_spectral',
    'discover_anchors_spectral_fixed_k',
    'find_k_by_eigengap',
    # Geometry
    'SenseDecomposition',
    'decompose',
    'print_report',
    'print_cross_word_summary',
    'collect_all_angles',
    'plot_word_dashboard',
    'plot_molecular_diagram',
    'plot_cross_word_comparison',
    'plot_angle_summary',
]

# Add distillation exports if available
if DISTILLATION_AVAILABLE:
    __all__.extend([
        'IVADistiller',
        'DistillationResult',
        'distill_concept',
        'measure_set_coherence',
        'validate_distillation',
        'create_distiller_from_explorer',
    ])

# Note: Merger functionality is available via:
#   from sense_explorer.merger import EmbeddingMerger, StagedMerger, ...
# This keeps the basic namespace clean while making advanced features accessible.
