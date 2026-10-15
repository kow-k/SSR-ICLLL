#!/usr/bin/env python3
"""
Render two radial dendrograms side by side for comparison.

Typical use: compare baseline (bare embeddings, one per word) vs.
SSR-treated (multiple sense vectors per word) on the same word list.

Usage:
    python compare_clusterings.py <left.json> <right.json> -o output.pdf \
        [--left-title TITLE] [--right-title TITLE] \
        [-k N] [--exclude-defaults] [--min-senses N] [--linkage METHOD]

Example:
    python compare_clusterings.py \
        bnc_verbs_120_baseline.json bnc_verbs_120.json \
        -o bnc_verbs_compare.pdf \
        --left-title "Baseline (no SSR)" \
        --right-title "With SSR" \
        -k 12 --exclude-defaults
"""

import argparse
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sense_cluster_radial import (
    load_sense_vectors,
    hierarchical_cluster,
    cut_tree,
    compute_tree_positions,
    draw_radial_branch,
    generate_cluster_colors,
    analyze_clusters,
    suggest_cluster_names,
)


def _filter_data(vectors, labels, exclude_defaults: bool, min_senses: int):
    """Apply the same filters as sense_cluster_radial.py."""
    if exclude_defaults:
        keep = [i for i, l in enumerate(labels) if not l.endswith("[default]")]
        vectors = vectors[keep]
        labels = [labels[i] for i in keep]
    
    if min_senses > 0:
        word_counts = Counter(l.split("[")[0] for l in labels)
        keep = [i for i, l in enumerate(labels)
                if word_counts[l.split("[")[0]] >= min_senses]
        vectors = vectors[keep]
        labels = [labels[i] for i in keep]
    
    return vectors, labels


def _draw_dendrogram_on_axis(ax, vectors, labels, n_clusters, linkage_method, 
                              title, label_fontsize=6, branch_linewidth=0.8):
    """Draw a radial dendrogram on the provided polar axes."""
    n_leaves = len(labels)
    
    # Cluster
    Z = hierarchical_cluster(vectors, linkage_method=linkage_method, metric='cosine')
    cluster_ids = cut_tree(Z, n_clusters) - 1
    
    n_clusters_actual = len(set(cluster_ids))
    cluster_colors = generate_cluster_colors(n_clusters_actual)
    
    positions, leaf_order = compute_tree_positions(Z, n_leaves)
    
    # Determine cluster of each internal node for branch coloring
    node_cluster = {i: cluster_ids[i] for i in range(n_leaves)}
    for merge_idx in range(Z.shape[0]):
        node_id = n_leaves + merge_idx
        left = int(Z[merge_idx, 0])
        right = int(Z[merge_idx, 1])
        if node_cluster.get(left, -1) == node_cluster.get(right, -1):
            node_cluster[node_id] = node_cluster[left]
        else:
            node_cluster[node_id] = -1
    
    # Draw branches recursively
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
        draw_radial_branch(ax, pos, positions[left], color=left_color, linewidth=branch_linewidth)
        draw_radial_branch(ax, pos, positions[right], color=right_color, linewidth=branch_linewidth)
        draw_subtree(left)
        draw_subtree(right)
    
    root_id = n_leaves + Z.shape[0] - 1
    draw_subtree(root_id)
    
    # Draw labels aligned with radial bars
    label_offset = 0.02
    for leaf_id in leaf_order:
        pos = positions[leaf_id]
        angle = pos['angle']
        leaf_radius = pos['radius']
        label = labels[leaf_id]
        cluster = cluster_ids[leaf_id]
        color = cluster_colors[cluster]
        
        theta_deg = np.degrees(angle) % 360
        visual_angle = (90 - theta_deg) % 360
        if 90 < visual_angle < 270:
            rotation = visual_angle - 180
            ha = 'right'
        else:
            rotation = visual_angle
            ha = 'left'
        
        ax.text(angle, leaf_radius + label_offset, label,
               ha=ha, va='center', fontsize=label_fontsize,
               rotation=rotation, rotation_mode='anchor',
               color=color, zorder=3)
    
    ax.set_ylim(0, 1.0 + 0.30)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    return cluster_ids, cluster_colors


def main():
    parser = argparse.ArgumentParser(
        description="Render two radial dendrograms side by side for comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("left", help="Path to LEFT JSON file (e.g., baseline).")
    parser.add_argument("right", help="Path to RIGHT JSON file (e.g., SSR-treated).")
    parser.add_argument("-o", "--output", default="comparison.pdf",
                        help="Output PDF path.")
    parser.add_argument("--left-title", default="Baseline",
                        help="Title for left dendrogram.")
    parser.add_argument("--right-title", default="With SSR",
                        help="Title for right dendrogram.")
    parser.add_argument("--suptitle", default=None,
                        help="Overall figure title.")
    parser.add_argument("-k", "--n-clusters", type=int, default=None,
                        help="Number of clusters (applied to both sides). Default: sqrt(n_senses).")
    parser.add_argument("--linkage", default="average",
                        choices=["average", "single", "complete", "weighted", "ward"])
    parser.add_argument("--exclude-defaults", action="store_true",
                        help="Exclude [default] entries (SSR failures).")
    parser.add_argument("--min-senses", type=int, default=0,
                        help="Exclude words with fewer than this many senses (applied to right only).")
    parser.add_argument("--figsize", type=float, default=22,
                        help="Figure side length (square ~ figsize x figsize/2 actual).")
    parser.add_argument("--label-fontsize", type=float, default=None,
                        help="Label font size. Default: auto.")
    
    args = parser.parse_args()
    
    # Load both
    print(f"Loading LEFT from {args.left}...")
    left_vec, left_lab = load_sense_vectors(args.left)
    print(f"  {len(left_lab)} sense vectors from {len(set(l.split('[')[0] for l in left_lab))} words")
    
    print(f"Loading RIGHT from {args.right}...")
    right_vec, right_lab = load_sense_vectors(args.right)
    print(f"  {len(right_lab)} sense vectors from {len(set(l.split('[')[0] for l in right_lab))} words")
    
    # Filter
    left_vec, left_lab = _filter_data(left_vec, left_lab, args.exclude_defaults, 0)
    right_vec, right_lab = _filter_data(right_vec, right_lab, args.exclude_defaults, args.min_senses)
    
    print(f"\nAfter filtering: LEFT={len(left_lab)}, RIGHT={len(right_lab)}")
    
    if len(left_lab) < 2 or len(right_lab) < 2:
        print("ERROR: not enough vectors to cluster.")
        sys.exit(1)
    
    # Determine k
    if args.n_clusters is not None:
        n_clusters = args.n_clusters
    else:
        n_clusters = max(5, int(np.sqrt(max(len(left_lab), len(right_lab)))))
    print(f"Using n_clusters = {n_clusters}")
    
    # Auto-scale fontsize based on the larger panel
    n_max = max(len(left_lab), len(right_lab))
    if args.label_fontsize is not None:
        fontsize = args.label_fontsize
    elif n_max <= 80:
        fontsize = 7
    elif n_max <= 200:
        fontsize = 5
    elif n_max <= 400:
        fontsize = 4
    else:
        fontsize = 3
    
    # Render
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(args.figsize, args.figsize / 2),
        subplot_kw={'projection': 'polar'},
    )
    
    print(f"\nRendering LEFT dendrogram ({args.left_title})...")
    _draw_dendrogram_on_axis(
        ax_l, left_vec, left_lab, n_clusters, args.linkage,
        title=args.left_title, label_fontsize=fontsize, branch_linewidth=0.7,
    )
    
    print(f"Rendering RIGHT dendrogram ({args.right_title})...")
    _draw_dendrogram_on_axis(
        ax_r, right_vec, right_lab, n_clusters, args.linkage,
        title=args.right_title, label_fontsize=fontsize, branch_linewidth=0.7,
    )
    
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
