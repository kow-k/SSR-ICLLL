#!/usr/bin/env python3
"""
Sense Geometry Module for SenseExplorer
========================================

Analyzes the geometric structure of separated senses: how sense vectors
are arranged around a polysemous word vector in embedding space.

Key finding: Inter-sense angles cluster around a characteristic scale
(median ~48° at 100d), formally analogous to molecular bond geometry.
Senses pack under competing constraints — context distinctness (repulsion)
vs. word identity (attraction) — yielding equilibrium angles that encode
information packing, not semantic relatedness.

Three-tier angular structure:
  - Synonyms / same-sense words:          < 30°
  - Different senses of the same word:    ~35–55° (median ~48°)
  - Unrelated words:                      ~90°

Capabilities:
  - decompose(): Linear decomposition w ≈ Σ αₘsₘ + ε
  - SenseDecomposition: Rich dataclass with angles, coefficients, territories
  - Visualization: Molecular diagrams, heatmaps, dashboards
  - Cross-word analysis: Statistical summaries across multiple words

Usage:
    # Via SenseExplorer (recommended)
    >>> se = SenseExplorer.from_glove("glove.6B.100d.txt")
    >>> decomp = se.localize_senses("bank")
    >>> print(decomp.variance_explained_total)  # R²

    # Standalone with pre-extracted sense vectors
    >>> from sense_explorer.geometry import decompose, SenseDecomposition
    >>> decomp = decompose("bank", word_vec, {"financial": vec1, "river": vec2})

    # Cross-word geometry analysis
    >>> results = se.analyze_geometry(["bank", "cell", "run"])

Reference:
    Kuroda & Claude (2026), "From sense mining to sense induction via
    simulated self-repair", Section 11: The geometry of separated senses.

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
"""

import numpy as np
from numpy.linalg import norm, lstsq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Visualization imports are deferred to avoid hard matplotlib dependency
# in the core analysis path.  Functions that need plotting will import
# matplotlib lazily and raise a clear error if it's missing.

__all__ = [
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


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SenseDecomposition:
    """
    Complete result of decomposing a word vector into sense components.

    The decomposition solves:  w ≈ α₁s₁ + α₂s₂ + ... + αₖsₖ + ε

    Attributes:
        word: The polysemous word
        word_vector: Original word embedding (d,)
        sense_labels: Sense names in order
        sense_vectors: Sense embeddings as rows (k, d)
        coefficients: Mixing weights α (k,)
        sense_contributions: α_i * s_i for each sense (k, d)
        residual: ε = w − ŵ (d,)
        reconstruction: ŵ = Σ α_i s_i (d,)
        variance_explained_total: R² = 1 − ||ε||² / ||w||²
        variance_explained_per_sense: Per-sense R² contributions (k,)
        residual_norm_ratio: ||ε|| / ||w||
        dimension_attribution: Fractional attribution per dim (k+1, d)
        dimension_dominance: Index of dominant sense per dim (d,)
        shared_dimensions: Boolean mask of shared dims (d,)
        unique_dimensions: Per-sense unique dim indices
        contested_dimensions: Boolean mask of sign-contested dims (d,)
        sense_cosine_matrix: Pairwise cosine similarities (k, k)
        sense_angles_deg: Pairwise angles in degrees (k, k)
        word_sense_cosines: cos(w, s_i) for each sense (k,)
        word_sense_projections: Projection of w onto each s_i (k,)
        sign_agreement: Whether α_i*s_i agrees with w in sign (k, d)
        constructive_dims: Per-sense dims that reinforce w
        destructive_dims: Per-sense dims that oppose w
    """
    word: str
    word_vector: np.ndarray
    sense_labels: List[str]
    sense_vectors: np.ndarray          # (k, d)
    coefficients: np.ndarray           # (k,)
    sense_contributions: np.ndarray    # (k, d)
    residual: np.ndarray               # (d,)
    reconstruction: np.ndarray         # (d,)
    variance_explained_total: float
    variance_explained_per_sense: np.ndarray  # (k,)
    residual_norm_ratio: float
    dimension_attribution: np.ndarray  # (k+1, d) — last row is residual
    dimension_dominance: np.ndarray    # (d,) int
    shared_dimensions: np.ndarray      # (d,) bool
    unique_dimensions: Dict[str, np.ndarray]
    contested_dimensions: np.ndarray   # (d,) bool
    sense_cosine_matrix: np.ndarray    # (k, k)
    sense_angles_deg: np.ndarray       # (k, k)
    word_sense_cosines: np.ndarray     # (k,)
    word_sense_projections: np.ndarray # (k,)
    sign_agreement: np.ndarray         # (k, d) bool
    constructive_dims: Dict[str, np.ndarray]
    destructive_dims: Dict[str, np.ndarray]

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def n_senses(self) -> int:
        """Number of senses."""
        return len(self.sense_labels)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return len(self.word_vector)

    @property
    def coefficient_ratio(self) -> float:
        """Ratio of largest to smallest coefficient (dominance measure)."""
        abs_c = np.abs(self.coefficients)
        return float(abs_c.max() / (abs_c.min() + 1e-10))

    @property
    def dominant_sense(self) -> str:
        """Label of the dominant sense (largest |α|)."""
        return self.sense_labels[int(np.argmax(np.abs(self.coefficients)))]

    @property
    def angle_pairs(self) -> List[Tuple[str, str, float]]:
        """All inter-sense (label_i, label_j, angle_deg) triples."""
        pairs = []
        k = self.n_senses
        for i in range(k):
            for j in range(i + 1, k):
                pairs.append((
                    self.sense_labels[i],
                    self.sense_labels[j],
                    float(self.sense_angles_deg[i, j])
                ))
        return pairs

    def summary_dict(self) -> dict:
        """Return a plain-dict summary suitable for JSON serialization."""
        return {
            'word': self.word,
            'n_senses': self.n_senses,
            'dim': self.dim,
            'r_squared': round(self.variance_explained_total, 4),
            'residual_ratio': round(self.residual_norm_ratio, 4),
            'coefficient_ratio': round(self.coefficient_ratio, 2),
            'dominant_sense': self.dominant_sense,
            'coefficients': {
                self.sense_labels[i]: round(float(self.coefficients[i]), 4)
                for i in range(self.n_senses)
            },
            'angles': {
                f"{s1}↔{s2}": round(angle, 1)
                for s1, s2, angle in self.angle_pairs
            },
        }


# =============================================================================
# Core Decomposition
# =============================================================================

def decompose(
    word: str,
    word_vector: np.ndarray,
    sense_dict: Dict[str, np.ndarray],
    sharing_threshold: float = 0.55
) -> SenseDecomposition:
    """
    Full decomposition of a word vector into sense components.

    Solves the linear system:  w = S^T α + ε
    where S is (k, d) sense vectors and α is (k,) coefficients.

    Args:
        word: The polysemous word
        word_vector: The word's embedding vector (d,)
        sense_dict: {sense_label: sense_vector} with vectors of shape (d,)
        sharing_threshold: Max attribution fraction for a dimension to
                          be considered "shared" (not owned by any sense)

    Returns:
        SenseDecomposition with full geometric analysis

    Example:
        >>> decomp = decompose("bank", embeddings["bank"],
        ...                    {"financial": vec1, "river": vec2})
        >>> print(f"R² = {decomp.variance_explained_total:.3f}")
        >>> for s1, s2, angle in decomp.angle_pairs:
        ...     print(f"  ∠({s1}, {s2}) = {angle:.1f}°")
    """
    w = word_vector.copy().astype(float)
    d = len(w)

    sense_labels = list(sense_dict.keys())
    k = len(sense_labels)
    S = np.array([sense_dict[label] for label in sense_labels], dtype=float)

    # ── 1. Linear decomposition via least-squares ───────────────────────
    alpha, _, _, _ = lstsq(S.T, w, rcond=None)
    reconstruction = S.T @ alpha
    residual = w - reconstruction
    sense_contributions = alpha[:, np.newaxis] * S  # (k, d)

    # ── 2. Variance analysis ────────────────────────────────────────────
    w_var = np.sum(w ** 2)
    r_var = np.sum(residual ** 2)
    var_explained_total = 1.0 - r_var / w_var if w_var > 0 else 0.0
    residual_norm_ratio = norm(residual) / norm(w) if norm(w) > 0 else 0.0
    var_per_sense = np.array([
        np.sum(sense_contributions[i] ** 2) / w_var if w_var > 0 else 0.0
        for i in range(k)
    ])

    # ── 3. Dimensional analysis ─────────────────────────────────────────
    abs_contrib = np.abs(sense_contributions)
    abs_resid = np.abs(residual)
    all_abs = np.vstack([abs_contrib, abs_resid[np.newaxis, :]])
    dim_totals = np.sum(all_abs, axis=0)
    dim_totals = np.where(dim_totals < 1e-10, 1.0, dim_totals)
    dimension_attribution = all_abs / dim_totals

    dimension_dominance = np.argmax(abs_contrib, axis=0)
    max_sense_attr = np.max(dimension_attribution[:k], axis=0)
    shared_dims = max_sense_attr < sharing_threshold

    unique_dims = {}
    for i, label in enumerate(sense_labels):
        is_dominant = (dimension_dominance == i) & (~shared_dims)
        unique_dims[label] = np.where(is_dominant)[0]

    contested = np.zeros(d, dtype=bool)
    if k >= 2:
        for di in range(d):
            signs = np.sign(sense_contributions[:, di])
            nonzero = signs[np.abs(sense_contributions[:, di]) > 1e-6]
            if len(nonzero) >= 2:
                contested[di] = (np.min(nonzero) < 0 and np.max(nonzero) > 0)

    # ── 4. Sense geometry ───────────────────────────────────────────────
    cosine_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ni, nj = norm(S[i]), norm(S[j])
            if ni > 0 and nj > 0:
                cosine_matrix[i, j] = np.dot(S[i], S[j]) / (ni * nj)
    angles_deg = np.degrees(np.arccos(np.clip(cosine_matrix, -1, 1)))

    w_norm = norm(w)
    word_sense_cosines = np.array([
        np.dot(w, S[i]) / (w_norm * norm(S[i]))
        if w_norm > 0 and norm(S[i]) > 0 else 0.0
        for i in range(k)
    ])
    word_sense_projections = np.array([
        np.dot(w, S[i]) / norm(S[i]) if norm(S[i]) > 0 else 0.0
        for i in range(k)
    ])

    # ── 5. Sign analysis ───────────────────────────────────────────────
    sign_agreement = np.sign(sense_contributions) == np.sign(w)[np.newaxis, :]
    near_zero = np.abs(w) < 1e-6
    sign_agreement[:, near_zero] = True

    constructive = {}
    destructive = {}
    for i, label in enumerate(sense_labels):
        significant = np.abs(sense_contributions[i]) > 1e-6
        constructive[label] = np.where(sign_agreement[i] & significant)[0]
        destructive[label] = np.where(~sign_agreement[i] & significant)[0]

    return SenseDecomposition(
        word=word, word_vector=w, sense_labels=sense_labels,
        sense_vectors=S, coefficients=alpha,
        sense_contributions=sense_contributions,
        residual=residual, reconstruction=reconstruction,
        variance_explained_total=var_explained_total,
        variance_explained_per_sense=var_per_sense,
        residual_norm_ratio=residual_norm_ratio,
        dimension_attribution=dimension_attribution,
        dimension_dominance=dimension_dominance,
        shared_dimensions=shared_dims, unique_dimensions=unique_dims,
        contested_dimensions=contested,
        sense_cosine_matrix=cosine_matrix,
        sense_angles_deg=angles_deg,
        word_sense_cosines=word_sense_cosines,
        word_sense_projections=word_sense_projections,
        sign_agreement=sign_agreement,
        constructive_dims=constructive, destructive_dims=destructive,
    )


# =============================================================================
# Batch / Cross-Word Utilities
# =============================================================================

def collect_all_angles(
    decompositions: List[SenseDecomposition]
) -> List[Tuple[str, str, str, float]]:
    """
    Collect every inter-sense angle across multiple decompositions.

    Returns:
        List of (word, sense_i, sense_j, angle_deg) tuples,
        sorted by angle.
    """
    all_angles = []
    for d in decompositions:
        for s1, s2, angle in d.angle_pairs:
            all_angles.append((d.word, s1, s2, angle))
    all_angles.sort(key=lambda x: x[3])
    return all_angles


# =============================================================================
# Text Reports
# =============================================================================

def print_report(d: SenseDecomposition) -> None:
    """Print a text summary of one word's decomposition."""
    k = d.n_senses
    print(f"\n{'=' * 70}")
    print(f"SENSE LOCALIZATION: '{d.word}'")
    print(f"{'=' * 70}")

    terms = [f"{d.coefficients[i]:.3f}·{d.sense_labels[i]}" for i in range(k)]
    print(f"\n  w = " + " + ".join(terms) + " + residual")

    print(f"\n  Variance Explained:")
    print(f"    Total R²: {d.variance_explained_total:.4f}")
    for i, label in enumerate(d.sense_labels):
        print(f"    {label}: {d.variance_explained_per_sense[i]:.4f}")
    print(f"    Residual ||r||/||w||: {d.residual_norm_ratio:.4f}")

    print(f"\n  Sense Geometry:")
    for i, label in enumerate(d.sense_labels):
        print(f"    cos(w, {label}) = {d.word_sense_cosines[i]:.4f}")
    for s1, s2, angle in d.angle_pairs:
        i = d.sense_labels.index(s1)
        j = d.sense_labels.index(s2)
        cos_val = d.sense_cosine_matrix[i, j]
        print(f"    ∠({s1}, {s2}) = {angle:.1f}°  (cos = {cos_val:.4f})")

    ndims = d.dim
    n_shared = int(np.sum(d.shared_dimensions))
    n_contested = int(np.sum(d.contested_dimensions))
    print(f"\n  Dimensional Territories ({ndims} dims):")
    print(f"    Shared (no clear owner): {n_shared} ({100 * n_shared / ndims:.0f}%)")
    print(f"    Contested (opposite signs): {n_contested} ({100 * n_contested / ndims:.0f}%)")
    for label in d.sense_labels:
        n_unique = len(d.unique_dimensions[label])
        n_cons = len(d.constructive_dims[label])
        n_dest = len(d.destructive_dims[label])
        print(f"    {label}: {n_unique} unique dims | "
              f"{n_cons} constructive | {n_dest} destructive")


def print_cross_word_summary(decompositions: List[SenseDecomposition]) -> None:
    """Print statistical generalizations across multiple words."""
    n = len(decompositions)
    print(f"\n{'=' * 70}")
    print(f"CROSS-WORD GENERALIZATIONS ({n} words)")
    print(f"{'=' * 70}")

    r2_vals = [d.variance_explained_total for d in decompositions]
    resid_vals = [d.residual_norm_ratio for d in decompositions]

    print(f"\n  Variance Explained:")
    print(f"    Mean R²: {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
    print(f"    Range: [{min(r2_vals):.4f}, {max(r2_vals):.4f}]")
    print(f"    Mean residual ratio: {np.mean(resid_vals):.4f}")

    all_angles = collect_all_angles(decompositions)
    if all_angles:
        angles_only = [a[3] for a in all_angles]
        print(f"\n  Inter-Sense Angles (THE KEY RESULT):")
        print(f"    Mean: {np.mean(angles_only):.1f}° ± {np.std(angles_only):.1f}°")
        print(f"    Median: {np.median(angles_only):.1f}°")
        print(f"    Range: [{min(angles_only):.1f}°, {max(angles_only):.1f}°]")
        near_orth = sum(1 for a in angles_only if 70 <= a <= 110)
        print(f"    Near-orthogonal (70-110°): {near_orth}/{len(angles_only)}")
        print(f"\n    Per word:")
        for word, s1, s2, angle in all_angles:
            print(f"      {word}: ∠({s1}, {s2}) = {angle:.1f}°")

    sharing_rates = [np.sum(d.shared_dimensions) / d.dim for d in decompositions]
    contested_rates = [np.sum(d.contested_dimensions) / d.dim for d in decompositions]
    print(f"\n  Dimensional Structure:")
    print(f"    Mean sharing rate: {np.mean(sharing_rates):.2%}")
    print(f"    Mean contested rate: {np.mean(contested_rates):.2%}")

    print(f"\n  Coefficient Patterns (sense mixing weights):")
    for d in decompositions:
        coefs = " | ".join(
            f"{d.sense_labels[i]}={d.coefficients[i]:.3f}"
            for i in range(d.n_senses)
        )
        print(f"    {d.word}: {coefs}")

    print(f"\n  Dominant Sense Bias:")
    for d in decompositions:
        print(f"    {d.word}: dominant = {d.dominant_sense} "
              f"({d.coefficient_ratio:.2f}x)")


# =============================================================================
# Visualization Helpers
# =============================================================================

# Default palette — the same ordering everywhere
SENSE_COLORS = [
    '#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
    '#00BCD4', '#795548', '#607D8B',
]
RESIDUAL_COLOR = '#BDBDBD'


def _import_matplotlib():
    """Lazy import of matplotlib; raises clear error if missing."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
        return plt, mpatches, LinearSegmentedColormap
    except ImportError:
        raise ImportError(
            "matplotlib is required for geometry visualization.\n"
            "Install it with: pip install matplotlib"
        )


def _import_pca():
    """Lazy import of PCA; raises clear error if missing."""
    try:
        from sklearn.decomposition import PCA
        return PCA
    except ImportError:
        raise ImportError(
            "scikit-learn is required for PCA-based visualization.\n"
            "Install it with: pip install scikit-learn"
        )


# =============================================================================
# Individual-Plot Functions
# =============================================================================

def plot_dimension_attribution(decomp, ax, sort_by_dominance=False, title=None):
    """Stacked bar chart of per-dimension sense attribution."""
    d = decomp.dim
    k = decomp.n_senses
    attr = decomp.dimension_attribution

    if sort_by_dominance:
        dom = decomp.dimension_dominance
        max_attr = np.max(attr[:k], axis=0)
        sort_idx = np.lexsort((-max_attr, dom))
        attr = attr[:, sort_idx]
        xlabel = "Dimensions (sorted by dominance)"
    else:
        xlabel = "Dimension index"

    x = np.arange(d)
    bottom = np.zeros(d)
    for i in range(k):
        ax.bar(x, attr[i], bottom=bottom, width=1.0,
               color=SENSE_COLORS[i % len(SENSE_COLORS)],
               label=decomp.sense_labels[i], edgecolor='none')
        bottom += attr[i]
    ax.bar(x, attr[k], bottom=bottom, width=1.0,
           color=RESIDUAL_COLOR, label='residual', edgecolor='none')

    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Attribution fraction")
    ax.set_title(title or f"Dimension Attribution: '{decomp.word}'")
    ax.legend(loc='upper right', fontsize=8)


def plot_sense_territories(decomp, ax, title=None):
    """Color-coded bar chart showing which sense owns each dimension."""
    import matplotlib.patches as mpatches

    d = decomp.dim
    k = decomp.n_senses

    categories = np.full(d, k + 1, dtype=int)
    for i, label in enumerate(decomp.sense_labels):
        for dim_idx in decomp.unique_dimensions[label]:
            categories[dim_idx] = i
    for dim_idx in range(d):
        if decomp.contested_dimensions[dim_idx]:
            categories[dim_idx] = k + 2

    sort_idx = np.argsort(categories)
    sorted_cats = categories[sort_idx]
    sorted_w = np.abs(decomp.word_vector[sort_idx])
    sorted_w = sorted_w / (np.max(sorted_w) + 1e-10)

    colors = []
    for cat in sorted_cats:
        if cat < k:
            colors.append(SENSE_COLORS[cat % len(SENSE_COLORS)])
        elif cat == k + 1:
            colors.append('#FFE082')
        else:
            colors.append('#EF5350')

    ax.bar(np.arange(d), sorted_w, width=1.0, color=colors, edgecolor='none')

    handles = []
    for i, label in enumerate(decomp.sense_labels):
        n = len(decomp.unique_dimensions[label])
        handles.append(mpatches.Patch(
            color=SENSE_COLORS[i % len(SENSE_COLORS)],
            label=f'{label} ({n})')
        )
    n_s = int(np.sum(decomp.shared_dimensions))
    n_c = int(np.sum(decomp.contested_dimensions))
    handles.append(mpatches.Patch(color='#FFE082', label=f'shared ({n_s})'))
    handles.append(mpatches.Patch(color='#EF5350', label=f'contested ({n_c})'))

    ax.set_xlim(-0.5, d - 0.5)
    ax.set_xlabel("Dimensions (sorted by territory)")
    ax.set_ylabel("|w[d]| (normalized)")
    ax.set_title(title or f"Sense Territories: '{decomp.word}'")
    ax.legend(handles=handles, loc='upper right', fontsize=8)


def plot_contribution_profiles(decomp, ax, title=None):
    """Overlay plot of each sense's per-dimension contribution."""
    d = decomp.dim
    x = np.arange(d)

    ax.bar(x, decomp.word_vector, width=1.0, color='#E0E0E0', alpha=0.5, label='w')
    for i, label in enumerate(decomp.sense_labels):
        ax.plot(x, decomp.sense_contributions[i],
                color=SENSE_COLORS[i % len(SENSE_COLORS)],
                linewidth=1.5, label=f'α·{label}', alpha=0.8)
    ax.plot(x, decomp.residual, color=RESIDUAL_COLOR, linewidth=1.0,
            linestyle='--', label='residual', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlim(-0.5, d - 0.5)
    ax.set_xlabel("Dimension index")
    ax.set_ylabel("Value")
    ax.set_title(title or f"Contribution Profiles: '{decomp.word}'")
    ax.legend(loc='upper right', fontsize=8)


def plot_sense_geometry(decomp, ax, title=None):
    """PCA-projected 2D diagram of word, sense, and reconstruction vectors."""
    PCA = _import_pca()
    k = decomp.n_senses
    vectors = np.vstack([
        decomp.word_vector[np.newaxis, :],
        decomp.sense_vectors,
        decomp.reconstruction[np.newaxis, :],
    ])

    pca = PCA(n_components=2)
    proj = pca.fit_transform(vectors)
    max_n = np.max(np.linalg.norm(proj, axis=1))
    if max_n > 0:
        proj = proj / max_n

    ax.annotate('', xy=proj[0], xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.text(proj[0, 0] * 1.08, proj[0, 1] * 1.08, 'w',
            fontsize=12, fontweight='bold', ha='center', va='center')

    for i in range(k):
        c = SENSE_COLORS[i % len(SENSE_COLORS)]
        ax.annotate('', xy=proj[1 + i], xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2.0))
        ax.text(proj[1 + i, 0] * 1.12, proj[1 + i, 1] * 1.12,
                decomp.sense_labels[i], fontsize=10, color=c,
                fontweight='bold', ha='center', va='center')

    ri = 1 + k
    ax.annotate('', xy=proj[ri], xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='grey', lw=1.5,
                                linestyle='dashed'))
    ax.text(proj[ri, 0] * 1.12, proj[ri, 1] * 1.12, 'ŵ',
            fontsize=10, color='grey', ha='center', va='center')

    for i in range(k):
        for j in range(i + 1, k):
            angle = decomp.sense_angles_deg[i, j]
            mid = (proj[1 + i] + proj[1 + j]) / 2
            mn = np.linalg.norm(mid)
            if mn > 0:
                mid = mid / mn * 0.35
            ax.text(mid[0], mid[1], f'{angle:.0f}°', fontsize=9,
                    ha='center', va='center', color='#666666',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#CCCCCC', alpha=0.8))

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='#EEEEEE', linewidth=0.5)
    ax.axvline(x=0, color='#EEEEEE', linewidth=0.5)
    ve = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ve[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({ve[1]:.0%} var)")
    ax.set_title(title or f"Sense Geometry: '{decomp.word}'")


def plot_sign_interference(decomp, ax, title=None):
    """Heatmap of constructive (blue) vs destructive (red) interference."""
    plt, _, LinearSegmentedColormap = _import_matplotlib()
    k = decomp.n_senses
    d = decomp.dim

    interference = np.zeros((k, d))
    for i in range(k):
        sig = np.abs(decomp.sense_contributions[i]) > 1e-6
        agrees = (np.sign(decomp.sense_contributions[i])
                  == np.sign(decomp.word_vector))
        interference[i, sig & agrees] = 1.0
        interference[i, sig & ~agrees] = -1.0

    cmap = LinearSegmentedColormap.from_list(
        'interf', ['#F44336', '#FFFFFF', '#2196F3'])
    im = ax.imshow(interference, aspect='auto', cmap=cmap, vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_yticks(range(k))
    ax.set_yticklabels(decomp.sense_labels)
    ax.set_xlabel("Dimension index")
    ax.set_title(title or f"Interference: '{decomp.word}' "
                 "(blue=construct, red=destruct)")
    plt.colorbar(im, ax=ax, shrink=0.8, ticks=[-1, 0, 1])


# =============================================================================
# Molecular-Style Diagram
# =============================================================================

def plot_molecular_diagram(decomp, ax, title=None):
    """
    Draw a molecular-style diagram: word vector at center, sense vectors
    radiating outward at their true inter-sense angles, arrow width ∝ α.

    For 2 senses the layout is exact; for k≥3 it uses MDS to preserve
    pairwise angles as well as possible in 2D.
    """
    import matplotlib.patches as patches

    k = decomp.n_senses
    angles = decomp.sense_angles_deg
    coefs = decomp.coefficients
    labels = decomp.sense_labels

    # ── Arrange senses in 2D preserving angles ─────────────────────────
    if k == 2:
        theta = np.radians(angles[0, 1])
        half = theta / 2
        coords = np.array([
            [np.cos(-half), np.sin(-half)],
            [np.cos(half), np.sin(half)],
        ])
    else:
        try:
            from sklearn.manifold import MDS
            dist = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    if i != j:
                        dist[i, j] = 1 - np.cos(np.radians(angles[i, j]))
            mds = MDS(n_components=2, dissimilarity='precomputed',
                      random_state=42, normalized_stress=False, max_iter=1000)
            coords = mds.fit_transform(dist)
            norms = np.linalg.norm(coords, axis=1, keepdims=True) + 1e-10
            coords = coords / norms
        except ImportError:
            # Fallback: equally-spaced angles
            angles_rad = np.linspace(0, 2 * np.pi, k, endpoint=False)
            coords = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])

    # ── Draw ───────────────────────────────────────────────────────────
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect('equal')
    ax.set_facecolor('#fafafa')

    # Center: word vector
    ax.plot(0, 0, 'ko', markersize=12, zorder=10)
    ax.text(0, -0.18, f'"{decomp.word}"', ha='center', va='top',
            fontsize=11, fontweight='bold', zorder=11)

    c_min, c_max = coefs.min(), coefs.max()
    arrow_length = 1.15
    dir_angles = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))

    for i in range(k):
        dx = coords[i, 0] * arrow_length
        dy = coords[i, 1] * arrow_length
        width = 1.5 + 3.5 * (coefs[i] - c_min) / (c_max - c_min + 1e-9)
        color = SENSE_COLORS[i % len(SENSE_COLORS)]
        ax.annotate('', xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=width, mutation_scale=15))
        label_r = arrow_length + 0.28
        lx = coords[i, 0] * label_r
        ly = coords[i, 1] * label_r
        ax.text(lx, ly, f'{labels[i]}\n(α={coefs[i]:.2f})',
                ha='center', va='center', fontsize=8.5,
                fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.9))

    # ── Angle arcs ─────────────────────────────────────────────────────
    arc_radii = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    pair_idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            arc_r = arc_radii[pair_idx % len(arc_radii)]
            a1, a2 = dir_angles[i], dir_angles[j]
            theta1, theta2 = min(a1, a2), max(a1, a2)
            if theta2 - theta1 > 180:
                theta1, theta2 = theta2, theta1 + 360
            arc = patches.Arc((0, 0), 2 * arc_r, 2 * arc_r, angle=0,
                              theta1=theta1, theta2=theta2,
                              color='#555555', lw=1.0, ls='-')
            ax.add_patch(arc)
            mid_angle = (a1 + a2) / 2
            if abs(a2 - a1) > 180:
                mid_angle += 180
            mid_rad = np.radians(mid_angle)
            mx = (arc_r + 0.08) * np.cos(mid_rad)
            my = (arc_r + 0.08) * np.sin(mid_rad)
            ax.text(mx, my, f'{angles[i, j]:.1f}°', ha='center', va='center',
                    fontsize=7.5, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffcc',
                              edgecolor='#cccc00', alpha=0.85))
            pair_idx += 1

    r2 = decomp.variance_explained_total
    ax.set_title(title or f'{decomp.word.upper()} ({decomp.dim}d, '
                 f'R² = {r2:.3f})', fontsize=12, fontweight='bold')
    ax.axis('off')


# =============================================================================
# Composite / Dashboard Figures
# =============================================================================

def plot_word_dashboard(
    decomp: SenseDecomposition,
    save_path: str,
    dpi: int = 150
) -> None:
    """
    Generate a full-page dashboard for one word's sense decomposition.

    Includes: attribution (2×), territories, contribution profiles,
    geometry PCA, sign interference, and a summary text box.
    """
    plt, _, _ = _import_matplotlib()

    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(6, 2, hspace=0.4, wspace=0.3,
                          height_ratios=[1, 1, 1, 1, 0.8, 0.6])

    ax1 = fig.add_subplot(gs[0, 0])
    plot_dimension_attribution(decomp, ax1, sort_by_dominance=False,
                               title="Attribution (original order)")
    ax2 = fig.add_subplot(gs[0, 1])
    plot_dimension_attribution(decomp, ax2, sort_by_dominance=True,
                               title="Attribution (sorted by dominance)")
    ax3 = fig.add_subplot(gs[1, :])
    plot_sense_territories(decomp, ax3)
    ax4 = fig.add_subplot(gs[2, :])
    plot_contribution_profiles(decomp, ax4)
    ax5 = fig.add_subplot(gs[3, 0])
    plot_sense_geometry(decomp, ax5)
    ax6 = fig.add_subplot(gs[3, 1])
    plot_molecular_diagram(decomp, ax6)
    ax7 = fig.add_subplot(gs[4, :])
    plot_sign_interference(decomp, ax7)

    # Summary text
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('off')
    k = decomp.n_senses
    lines = [
        f"Word: '{decomp.word}'   |   Senses: {k}   |   Dims: {decomp.dim}",
        f"R² = {decomp.variance_explained_total:.4f}   |   "
        f"Residual ratio = {decomp.residual_norm_ratio:.4f}   |   "
        f"Coefficient ratio = {decomp.coefficient_ratio:.1f}:1",
        "",
        "Coefficients: " + "  |  ".join(
            f"α({decomp.sense_labels[i]}) = {decomp.coefficients[i]:.3f}"
            for i in range(k)),
        "Cosines: " + "  |  ".join(
            f"cos(w,{decomp.sense_labels[i]}) = "
            f"{decomp.word_sense_cosines[i]:.3f}"
            for i in range(k)),
    ]
    angle_strs = [f"∠({s1},{s2}) = {angle:.1f}°"
                  for s1, s2, angle in decomp.angle_pairs]
    if angle_strs:
        lines.append("Angles: " + "  |  ".join(angle_strs))

    n_shared = int(np.sum(decomp.shared_dimensions))
    n_contested = int(np.sum(decomp.contested_dimensions))
    terr = [f"{l}: {len(decomp.unique_dimensions[l])} unique"
            for l in decomp.sense_labels]
    lines.append(f"Territories: " + "  |  ".join(terr)
                 + f"  |  shared: {n_shared}  |  contested: {n_contested}")

    ax8.text(0.02, 0.95, "\n".join(lines), transform=ax8.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5',
                       edgecolor='#CCCCCC'))

    fig.suptitle(f"Sense Localization Dashboard: '{decomp.word}'",
                 fontsize=16, fontweight='bold', y=0.99)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_cross_word_comparison(
    decompositions: List[SenseDecomposition],
    save_path: str,
    dpi: int = 150
) -> None:
    """Side-by-side attribution + territories + geometry for multiple words."""
    plt, _, _ = _import_matplotlib()
    n = len(decompositions)
    fig, axes = plt.subplots(n, 3, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, decomp in enumerate(decompositions):
        plot_dimension_attribution(decomp, axes[row, 0],
                                   sort_by_dominance=True,
                                   title=f"'{decomp.word}' — Attribution")
        plot_sense_territories(decomp, axes[row, 1],
                               title=f"'{decomp.word}' — Territories")
        plot_sense_geometry(decomp, axes[row, 2],
                            title=f"'{decomp.word}' — Geometry")

    fig.suptitle("Cross-Word Sense Localization Comparison",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_angle_summary(
    decompositions: List[SenseDecomposition],
    save_path: str,
    dpi: int = 150
) -> None:
    """Bar chart of every inter-sense angle — the key empirical result."""
    plt, _, _ = _import_matplotlib()

    pairs = collect_all_angles(decompositions)
    if not pairs:
        return
    labels = [f"{w}\n({s1}/{s2})" for w, s1, s2, _ in pairs]
    angles = [a for _, _, _, a in pairs]

    fig, ax = plt.subplots(figsize=(max(12, len(pairs) * 1.2), 6))

    colors = ['#4CAF50' if 70 <= a <= 110 else
              ('#2196F3' if 35 <= a <= 55 else '#F44336') for a in angles]
    bars = ax.bar(range(len(pairs)), angles, color=colors,
                  edgecolor='white', linewidth=0.5)

    # Reference lines
    ax.axhspan(35, 55, alpha=0.08, color='blue',
               label='Sense regime (35-55°)')
    ax.axhspan(70, 110, alpha=0.06, color='green',
               label='Near-orthogonal (70-110°)')

    mean_angle = np.mean(angles)
    ax.axhline(y=mean_angle, color='red', linewidth=1.5, linestyle='--',
               label=f'Mean = {mean_angle:.1f}°')

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel("Inter-sense angle (degrees)")
    ax.set_title("Inter-Sense Angle Distribution",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(angles) + 15)
    ax.legend(loc='upper right')

    for i, (bar, angle) in enumerate(zip(bars, angles)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{angle:.0f}°', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
