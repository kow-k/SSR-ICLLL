#!/usr/bin/env python3
"""
Radial dendrogram visualization of word sense vectors via hierarchical clustering.

This script:
1. Loads sense vectors from SSR output
2. Performs hierarchical clustering on cosine distances
3. Renders a circular dendrogram with radial+arc branches

Author: Kow Kuroda (with Claude assistance)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import colorsys
from typing import Dict, List, Tuple, Optional
import json


def load_sense_vectors(filepath: str) -> Tuple[np.ndarray, List[str]]:
    """Load sense vectors from JSON: {"word": {"sense": [vector], ...}, ...}"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    vectors, labels = [], []
    for word, senses in data.items():
        for sense_name, vector in senses.items():
            vectors.append(vector)
            labels.append(f"{word}[{sense_name}]")
    return np.array(vectors), labels


def hierarchical_cluster(vectors: np.ndarray, 
                          linkage_method: str = 'average',
                          metric: str = 'cosine') -> np.ndarray:
    """Hierarchical clustering. Returns linkage matrix (scipy format)."""
    if metric == 'cosine':
        distances = cosine_distances(vectors)
        condensed = squareform(distances, checks=False)
    else:
        condensed = pdist(vectors, metric=metric)
    
    if linkage_method == 'ward' and metric == 'cosine':
        print("Warning: Ward requires Euclidean. Using 'average' for cosine.")
        linkage_method = 'average'
    
    return linkage(condensed, method=linkage_method)


def cut_tree(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut dendrogram into flat clusters."""
    return fcluster(Z, t=n_clusters, criterion='maxclust')


def generate_cluster_colors(n_clusters: int) -> List[str]:
    """Generate visually distinct colors."""
    colors = []
    for i in range(n_clusters):
        hue = i / n_clusters
        rgb = colorsys.hls_to_rgb(hue, 0.5, 0.7)
        colors.append('#{:02x}{:02x}{:02x}'.format(
            int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
        ))
    return colors


def compute_tree_positions(Z: np.ndarray, n_leaves: int) -> Tuple[Dict, List[int]]:
    """
    Compute polar positions for all nodes in the dendrogram.
    
    Returns:
        positions: dict mapping node_id -> {angle, radius, is_leaf, ...}
        leaf_order: list of leaf IDs in circular order (left-to-right traversal)
    """
    n_internal = Z.shape[0]
    total_nodes = n_leaves + n_internal
    
    # Get leaf order by traversing tree
    def get_leaves(node_id):
        if node_id < n_leaves:
            return [node_id]
        merge_idx = node_id - n_leaves
        left = int(Z[merge_idx, 0])
        right = int(Z[merge_idx, 1])
        return get_leaves(left) + get_leaves(right)
    
    root_id = total_nodes - 1
    leaf_order = get_leaves(root_id)
    
    # Position leaves evenly around circle
    positions = {}
    angle_step = 2 * np.pi / n_leaves
    
    for i, leaf_id in enumerate(leaf_order):
        positions[leaf_id] = {
            'angle': i * angle_step,
            'radius': 1.0,
            'is_leaf': True,
            'merge_height': 0.0
        }
    
    # Position internal nodes
    max_height = Z[:, 2].max() if len(Z) > 0 else 1.0
    
    for merge_idx in range(n_internal):
        node_id = n_leaves + merge_idx
        left = int(Z[merge_idx, 0])
        right = int(Z[merge_idx, 1])
        height = Z[merge_idx, 2]
        
        left_angle = positions[left]['angle']
        right_angle = positions[right]['angle']
        
        # Take midpoint (handling wraparound)
        diff = right_angle - left_angle
        if abs(diff) > np.pi:
            # Wraparound case
            mid_angle = ((left_angle + right_angle) / 2 + np.pi) % (2 * np.pi)
        else:
            mid_angle = (left_angle + right_angle) / 2
        
        # Inner nodes closer to center (higher merges = closer to root)
        normalized_height = height / max_height
        radius = 1.0 - normalized_height * 0.85
        
        positions[node_id] = {
            'angle': mid_angle,
            'radius': radius,
            'is_leaf': False,
            'merge_height': height,
            'left': left,
            'right': right
        }
    
    return positions, leaf_order


def draw_radial_branch(ax, parent_pos: dict, child_pos: dict, 
                       color: str = '#333333', linewidth: float = 1.0):
    """
    Draw a branch as: arc at parent radius + radial line to child.
    This creates the characteristic dendrogram shape.
    """
    parent_angle = parent_pos['angle']
    parent_radius = parent_pos['radius']
    child_angle = child_pos['angle']
    child_radius = child_pos['radius']
    
    # Shortest arc direction
    diff = child_angle - parent_angle
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    
    # Arc segment at parent's radius (from parent angle to child angle)
    if abs(diff) > 1e-6:
        n_points = max(10, int(abs(diff) * 60))
        arc_angles = np.linspace(parent_angle, parent_angle + diff, n_points)
        ax.plot(arc_angles, [parent_radius] * n_points, 
               color=color, linewidth=linewidth,
               solid_capstyle='round', zorder=2)
    
    # Radial segment from parent's radius to child's radius
    ax.plot([child_angle, child_angle], [parent_radius, child_radius],
           color=color, linewidth=linewidth, 
           solid_capstyle='round', zorder=2)


def create_radial_dendrogram(vectors: np.ndarray,
                              labels: List[str],
                              n_clusters: Optional[int] = None,
                              linkage_method: str = 'average',
                              cluster_labels: Optional[np.ndarray] = None,
                              cluster_names: Optional[Dict[int, str]] = None,
                              figsize: Tuple[int, int] = (16, 16),
                              title: str = "Hierarchical Clustering of Word Senses",
                              output_path: Optional[str] = None,
                              label_fontsize: int = 7,
                              branch_linewidth: float = 0.8,
                              show_labels: bool = True,
                              label_margin: float = 0.30,
                              show_title: bool = True,
                              legend_min_size: int = 0) -> plt.Figure:
    """
    Create a circular dendrogram from sense vectors.
    
    The tree branches use radial+arc segments (characteristic dendrogram shape).
    Leaves on outer circle; internal nodes inward; root near center.
    """
    n_leaves = len(labels)
    
    # Hierarchical clustering
    print(f"Hierarchical clustering ({linkage_method} linkage, cosine distance)...")
    Z = hierarchical_cluster(vectors, linkage_method=linkage_method, metric='cosine')
    
    # Cluster assignments for coloring
    if cluster_labels is None:
        if n_clusters is None:
            n_clusters = max(5, int(np.sqrt(n_leaves)))
        cluster_labels = cut_tree(Z, n_clusters) - 1
    
    n_clusters_actual = len(set(cluster_labels))
    cluster_colors = generate_cluster_colors(n_clusters_actual)
    
    # Compute node positions
    positions, leaf_order = compute_tree_positions(Z, n_leaves)
    
    # Determine cluster for each internal node (for branch coloring)
    node_cluster = {i: cluster_labels[i] for i in range(n_leaves)}
    
    for merge_idx in range(Z.shape[0]):
        node_id = n_leaves + merge_idx
        left = int(Z[merge_idx, 0])
        right = int(Z[merge_idx, 1])
        if node_cluster.get(left, -1) == node_cluster.get(right, -1):
            node_cluster[node_id] = node_cluster[left]
        else:
            node_cluster[node_id] = -1  # Mixed cluster
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Draw tree (recursive from root)
    def draw_subtree(node_id):
        pos = positions[node_id]
        if pos['is_leaf']:
            return
        
        left = pos['left']
        right = pos['right']
        
        left_cluster = node_cluster.get(left, -1)
        right_cluster = node_cluster.get(right, -1)
        
        left_color = cluster_colors[left_cluster] if left_cluster >= 0 else '#888888'
        right_color = cluster_colors[right_cluster] if right_cluster >= 0 else '#888888'
        
        draw_radial_branch(ax, pos, positions[left], 
                          color=left_color, linewidth=branch_linewidth)
        draw_radial_branch(ax, pos, positions[right], 
                          color=right_color, linewidth=branch_linewidth)
        
        draw_subtree(left)
        draw_subtree(right)
    
    root_id = n_leaves + Z.shape[0] - 1
    draw_subtree(root_id)
    
    # Draw leaf labels at the tips of radial branches
    # Each label is rotated to match its radial bar's visual angle.
    # 
    # Note: We use set_theta_zero_location('N') and set_theta_direction(-1) (clockwise).
    # The data angle theta is measured clockwise from North.
    # The visual angle (standard math convention: counterclockwise from East) is:
    #     visual_angle = 90 - theta_clockwise_from_N
    # The text 'rotation' parameter uses the standard math convention.
    label_offset = 0.02
    
    for leaf_id in (leaf_order if show_labels else []):
        pos = positions[leaf_id]
        angle = pos['angle']  # radians, clockwise from N
        leaf_radius = pos['radius']
        label = labels[leaf_id]
        cluster = cluster_labels[leaf_id]
        color = cluster_colors[cluster]
        
        # Convert to visual angle (degrees, counterclockwise from East)
        theta_deg = np.degrees(angle) % 360
        visual_angle = (90 - theta_deg) % 360
        
        # Text rotation = visual angle (so text runs along the radial)
        # Flip 180° if text would be upside down
        if 90 < visual_angle < 270:
            rotation = visual_angle - 180
            ha = 'right'  # text extends inward visually, so right-align at anchor
        else:
            rotation = visual_angle
            ha = 'left'   # text extends outward visually, so left-align at anchor
        
        ax.text(angle, leaf_radius + label_offset, label,
               ha=ha, va='center', fontsize=label_fontsize,
               rotation=rotation, rotation_mode='anchor',
               color=color, zorder=3)
    
    # Configure axes
    ax.set_ylim(0, 1.0 + label_margin)  # Leaf radius (1.0) + optional label ring
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Title (optional)
    if show_title and title:
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Legend
    if cluster_names:
        legend_elements = [
            mpatches.Patch(facecolor=cluster_colors[cid],
                          label=f"{cluster_names.get(cid, f'Cluster {cid}')} "
                                f"({np.sum(cluster_labels == cid)})")
            for cid in sorted(set(cluster_labels))
        ]
    else:
        legend_elements = [
            mpatches.Patch(facecolor=cluster_colors[cid],
                          label=f"Cluster {cid} ({np.sum(cluster_labels == cid)})")
            for cid in sorted(set(cluster_labels))
        ]
    
    if legend_min_size > 0:
        legend_elements = [
            le for le, cid in zip(legend_elements, sorted(set(cluster_labels)))
            if np.sum(cluster_labels == cid) >= legend_min_size
        ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left',
                 bbox_to_anchor=(-0.02, 1.02), fontsize=11, frameon=False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Saved figure to {output_path}")
    
    return fig


def analyze_clusters(labels: List[str],
                     cluster_ids: np.ndarray,
                     vectors: np.ndarray) -> Dict:
    """Analyze cluster composition and quality."""
    vectors_norm = normalize(vectors)
    analysis = {}
    
    for cid in sorted(set(cluster_ids)):
        mask = cluster_ids == cid
        cluster_labels_list = [l for l, m in zip(labels, mask) if m]
        cluster_vectors = vectors_norm[mask]
        
        if len(cluster_vectors) > 1:
            sim_matrix = cosine_similarity(cluster_vectors)
            upper_tri = sim_matrix[np.triu_indices(len(cluster_vectors), k=1)]
            mean_sim = upper_tri.mean() if len(upper_tri) > 0 else 1.0
        else:
            mean_sim = 1.0
        
        words = [l.split('[')[0] for l in cluster_labels_list]
        analysis[cid] = {
            'size': len(cluster_labels_list),
            'members': cluster_labels_list,
            'unique_words': len(set(words)),
            'mean_similarity': mean_sim,
        }
    return analysis


def print_cluster_report(analysis: Dict):
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS REPORT")
    print("="*70)
    for cid, info in sorted(analysis.items()):
        print(f"\n--- Cluster {cid} ({info['size']} senses) ---")
        print(f"    Within-cluster similarity: {info['mean_similarity']:.3f}")
        print(f"    Members: {', '.join(info['members'][:8])}", end='')
        if len(info['members']) > 8:
            print(f" ... (+{len(info['members'])-8} more)")
        else:
            print()


def suggest_cluster_names(analysis: Dict) -> Dict[int, str]:
    """Suggest cluster names from member word content."""
    domain_keywords = {
        'animal': ['animal', 'bird', 'fish', 'mammal', 'insect', 'creature', 'rodent'],
        'body': ['body', 'organ', 'limb', 'head', 'foot', 'hand', 'blood'],
        'technology': ['device', 'computer', 'digital', 'electronic', 'phone', 'circuit'],
        'institution': ['organization', 'company', 'government', 'legal', 'financial', 'committee'],
        'nature': ['river', 'water', 'mountain', 'forest', 'land', 'weather', 'flow'],
        'action': ['move', 'run', 'take', 'break', 'hit', 'strike'],
        'food': ['food', 'eat', 'meal', 'taste', 'sweet'],
        'material': ['metal', 'wood', 'stone', 'fabric', 'cloth'],
    }
    
    names = {}
    for cid, info in analysis.items():
        text = ' '.join(info['members']).lower()
        scores = {d: sum(1 for kw in kws if kw in text) for d, kws in domain_keywords.items()}
        scores = {d: s for d, s in scores.items() if s > 0}
        names[cid] = max(scores, key=scores.get).title() if scores else f"Cluster {cid}"
    return names


# =============================================================================
# DEMO
# =============================================================================

def create_demo_data() -> Tuple[np.ndarray, List[str]]:
    """Create synthetic demo data with 5 semantic clusters."""
    np.random.seed(42)
    dim = 100
    
    domains = {
        0: [  # Animals
            "bat[animal]", "crane[bird]", "mouse[rodent]", "fly[insect]",
            "bass[fish]", "perch[fish]", "sole[fish]", "hawk[bird]",
            "dove[bird]", "swallow[bird]", "mole[animal]", "seal[mammal]",
            "spider[insect]", "leech[worm]"
        ],
        1: [  # Technology
            "mouse[device]", "chip[microchip]", "board[circuit]", "screen[display]",
            "monitor[screen]", "drive[storage]", "server[computer]", "terminal[computer]",
            "cell[phone]", "tablet[device]", "web[internet]", "net[internet]"
        ],
        2: [  # Institutions
            "bank[financial]", "court[legal]", "bar[legal]", "cabinet[government]",
            "party[political]", "firm[company]", "trust[organization]", "board[committee]",
            "chair[position]", "office[workplace]", "union[organization]"
        ],
        3: [  # Nature
            "bank[river]", "spring[water]", "delta[river]", "bed[river]",
            "mouth[river]", "channel[waterway]", "current[flow]", "basin[geological]",
            "bay[water]", "cape[landform]", "shore[coastline]"
        ],
        4: [  # Body
            "head[body]", "foot[body]", "arm[body]", "eye[body]",
            "heart[organ]", "vessel[blood]", "chest[body]", "temple[head]",
            "palm[hand]", "crown[head]", "spine[body]", "joint[body]"
        ]
    }
    
    cluster_centers = normalize(np.random.randn(len(domains), dim))
    vectors, labels = [], []
    for cid, sense_labels in domains.items():
        center = cluster_centers[cid]
        for label in sense_labels:
            noise = np.random.randn(dim) * 0.3
            vectors.append(center + noise)
            labels.append(label)
    
    return np.array(vectors), labels


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Radial dendrogram visualization of word sense vectors via hierarchical clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input/output
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to sense vectors JSON file. If omitted, runs on synthetic demo data.")
    parser.add_argument("-o", "--output", default="sense_clusters_radial.pdf",
                        help="Output PDF path.")
    parser.add_argument("--title", default=None,
                        help="Plot title. Default: auto-generated from sense count.")
    
    # Clustering parameters
    parser.add_argument("-k", "--n-clusters", type=int, default=None,
                        help="Number of color-coded clusters. Default: sqrt(n_senses), min 5.")
    parser.add_argument("--linkage", default="average",
                        choices=["average", "single", "complete", "weighted", "ward"],
                        help="Hierarchical linkage method. Note: 'ward' requires Euclidean distance.")
    parser.add_argument("--metric", default="cosine",
                        choices=["cosine", "euclidean"],
                        help="Distance metric for clustering.")
    
    # Filtering
    parser.add_argument("--exclude-defaults", action="store_true",
                        help="Exclude entries whose sense name is 'default' (SSR-failure cases).")
    parser.add_argument("--min-senses", type=int, default=0,
                        help="Exclude words that produced fewer than this many senses.")
    
    # Visual parameters
    parser.add_argument("--figsize", type=float, default=None,
                        help="Figure side length in inches (square). Default: auto-scaled to sense count.")
    parser.add_argument("--label-fontsize", type=float, default=None,
                        help="Font size for leaf labels. Default: auto-scaled.")
    parser.add_argument("--branch-linewidth", type=float, default=None,
                        help="Width of dendrogram branches. Default: auto-scaled.")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide leaf (sense) labels for a compact figure.")
    parser.add_argument("--no-title", action="store_true",
                        help="Omit the plot title (use when a figure caption supplies it).")
    parser.add_argument("--label-margin", type=float, default=0.30,
                        help="Radial space reserved beyond the leaf circle for labels.")
    parser.add_argument("--legend-min-size", type=int, default=0,
                        help="Only show clusters with at least this many members in the legend.")
    
    args = parser.parse_args()
    
    print("="*70)
    print("RADIAL DENDROGRAM OF WORD SENSE VECTORS")
    print("="*70)
    
    # ---- Load data ----
    if args.input:
        print(f"\nLoading sense vectors from {args.input}...")
        vectors, labels = load_sense_vectors(args.input)
        n_words = len(set(l.split("[")[0] for l in labels))
        print(f"Loaded {len(labels)} sense vectors from {n_words} words")
        default_title = f"Hierarchical Clustering of {{n}} Word Senses"
    else:
        print("\nNo input file provided. Running with synthetic demo data.")
        print("Generating demo data...")
        vectors, labels = create_demo_data()
        print(f"Loaded {len(labels)} sense vectors")
        default_title = "Hierarchical Clustering of Word Senses (Demo)"
    
    # ---- Filter ----
    if args.exclude_defaults:
        keep = [i for i, l in enumerate(labels) if not l.endswith("[default]")]
        dropped = len(labels) - len(keep)
        if dropped:
            print(f"\nExcluding {dropped} default (SSR-failure) entries...")
            vectors = vectors[keep]
            labels = [labels[i] for i in keep]
    
    if args.min_senses > 0:
        # Count senses per word
        from collections import Counter
        word_counts = Counter(l.split("[")[0] for l in labels)
        keep = [i for i, l in enumerate(labels)
                if word_counts[l.split("[")[0]] >= args.min_senses]
        dropped = len(labels) - len(keep)
        if dropped:
            print(f"Excluding {dropped} entries from words with < {args.min_senses} senses...")
            vectors = vectors[keep]
            labels = [labels[i] for i in keep]
    
    if len(labels) < 2:
        print(f"ERROR: not enough sense vectors to cluster (n={len(labels)}).")
        sys.exit(1)
    
    print(f"\nClustering {len(labels)} sense vectors.")
    
    # ---- Determine n_clusters ----
    if args.n_clusters is not None:
        n_clusters = args.n_clusters
    elif args.input:
        n_clusters = max(5, int(np.sqrt(len(labels))))
    else:
        n_clusters = 5
    print(f"Using n_clusters = {n_clusters}")
    
    # ---- Cluster ----
    Z = hierarchical_cluster(vectors, linkage_method=args.linkage, metric=args.metric)
    cluster_ids = cut_tree(Z, n_clusters) - 1
    
    print("\nAnalyzing cluster composition...")
    analysis = analyze_clusters(labels, cluster_ids, vectors)
    print_cluster_report(analysis)
    
    cluster_names = suggest_cluster_names(analysis)
    print("\nSuggested cluster names:")
    for cid, name in cluster_names.items():
        print(f"  Cluster {cid} -> {name}")
    
    # ---- Visual auto-scaling ----
    n_leaves = len(labels)
    if n_leaves <= 80:
        auto_figsize, auto_fontsize, auto_lw = 16, 8, 1.0
    elif n_leaves <= 200:
        auto_figsize, auto_fontsize, auto_lw = 20, 6, 0.8
    elif n_leaves <= 400:
        auto_figsize, auto_fontsize, auto_lw = 24, 5, 0.6
    else:
        auto_figsize, auto_fontsize, auto_lw = 28, 4, 0.5
    
    fig_side = args.figsize if args.figsize is not None else auto_figsize
    figsize = (fig_side, fig_side)
    label_fontsize = args.label_fontsize if args.label_fontsize is not None else auto_fontsize
    branch_lw = args.branch_linewidth if args.branch_linewidth is not None else auto_lw
    
    # ---- Title ----
    if args.title is not None:
        title = args.title
    elif args.input:
        title = default_title.format(n=n_leaves)
    else:
        title = default_title
    
    print(f"\nGenerating radial dendrogram ({n_leaves} leaves, figsize={figsize})...")
    fig = create_radial_dendrogram(
        vectors=vectors,
        labels=labels,
        cluster_labels=cluster_ids,
        cluster_names=cluster_names,
        linkage_method=args.linkage,
        figsize=figsize,
        title=title,
        output_path=args.output,
        label_fontsize=label_fontsize,
        branch_linewidth=branch_lw,
        show_labels=not args.no_labels,
        label_margin=(0.03 if args.no_labels else args.label_margin),
        show_title=not args.no_title,
        legend_min_size=args.legend_min_size,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    import sys
    main()
