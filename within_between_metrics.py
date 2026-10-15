#!/usr/bin/env python3
"""
Quantify SSR's geometric effect by comparing within-word vs. between-word
distances in baseline and SSR-treated sense vectors.

The hypothesis: SSR pulls a word's senses apart from each other (creating
within-word distance) while preserving their location within the global
semantic neighborhood (between-word distance stays similar).

For each polysemous word in the SSR-treated JSON, we compute:
  d_within   = mean cosine distance among the word's own sense vectors
  d_between  = mean cosine distance from this word's sense vectors to all
               other words' sense vectors
  ratio      = d_within / d_between

Baseline has d_within = 0 for every word (one vector each), so the metric
is meaningful only for the SSR-treated condition. For comparison purposes,
we also report the *cross-word* distance statistics in baseline (the only
non-zero quantity baseline has).

Usage:
    python within_between_metrics.py <baseline_json> <ssr_json>
    python within_between_metrics.py bnc_verbs_120_baseline.json bnc_verbs_120.json
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def load_grouped(json_path: str) -> Dict[str, np.ndarray]:
    """
    Load a sense-vectors JSON and return a dict mapping word -> matrix
    of shape (n_senses, dim).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    grouped = {}
    for word, senses in data.items():
        vectors = np.array([senses[s] for s in senses], dtype=np.float32)
        grouped[word] = vectors
    return grouped


def filter_defaults(grouped: Dict[str, np.ndarray], 
                    sense_keys: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """Remove words whose only sense is 'default' (SSR failures)."""
    out = {}
    for word, mat in grouped.items():
        keys = sense_keys[word]
        if len(keys) == 1 and keys[0] == 'default':
            continue
        out[word] = mat
    return out


def load_with_keys(json_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Load and also return the sense-key names per word."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    grouped = {}
    keys = {}
    for word, senses in data.items():
        sense_names = list(senses.keys())
        vectors = np.array([senses[s] for s in sense_names], dtype=np.float32)
        grouped[word] = vectors
        keys[word] = sense_names
    return grouped, keys


def compute_within(matrix: np.ndarray) -> float:
    """Mean cosine distance among rows of matrix. Returns 0.0 if <2 rows."""
    if matrix.shape[0] < 2:
        return 0.0
    D = cosine_distances(matrix)
    iu = np.triu_indices(matrix.shape[0], k=1)
    return float(D[iu].mean())


def compute_between(target: np.ndarray, others: np.ndarray) -> float:
    """Mean cosine distance from rows in target to all rows in others."""
    if target.shape[0] == 0 or others.shape[0] == 0:
        return float('nan')
    D = cosine_distances(target, others)
    return float(D.mean())


def per_word_metrics(grouped: Dict[str, np.ndarray]) -> List[dict]:
    """
    Compute per-word within/between metrics.
    
    For each word w:
      - d_within(w) = mean pairwise distance among w's sense vectors
      - d_between(w) = mean distance from w's senses to all other words' senses
    """
    # Build the "others" pool incrementally for efficiency
    all_words = list(grouped.keys())
    
    # Collect all vectors and their word labels
    all_vectors = []
    all_word_for_vector = []
    for w in all_words:
        for row in grouped[w]:
            all_vectors.append(row)
            all_word_for_vector.append(w)
    all_vectors = np.array(all_vectors, dtype=np.float32)
    all_word_for_vector = np.array(all_word_for_vector)
    
    rows = []
    for w in all_words:
        target = grouped[w]
        n_senses = target.shape[0]
        
        # Within-word distance
        d_w = compute_within(target)
        
        # Between-word distance: target vs. all rows from other words
        mask = all_word_for_vector != w
        others = all_vectors[mask]
        d_b = compute_between(target, others)
        
        rows.append({
            'word': w,
            'n_senses': n_senses,
            'd_within': d_w,
            'd_between': d_b,
            'ratio': d_w / d_b if d_b > 0 else float('nan'),
        })
    return rows


def summary_stats(rows: List[dict], filter_polysemous: bool = False) -> dict:
    """Compute aggregate statistics over per-word rows."""
    if filter_polysemous:
        rows = [r for r in rows if r['n_senses'] >= 2]
    
    if not rows:
        return {'n': 0}
    
    d_within = [r['d_within'] for r in rows]
    d_between = [r['d_between'] for r in rows if not np.isnan(r['d_between'])]
    ratios = [r['ratio'] for r in rows if not np.isnan(r['ratio'])]
    
    return {
        'n': len(rows),
        'mean_d_within': float(np.mean(d_within)) if d_within else float('nan'),
        'median_d_within': float(np.median(d_within)) if d_within else float('nan'),
        'mean_d_between': float(np.mean(d_between)) if d_between else float('nan'),
        'median_d_between': float(np.median(d_between)) if d_between else float('nan'),
        'mean_ratio': float(np.mean(ratios)) if ratios else float('nan'),
        'median_ratio': float(np.median(ratios)) if ratios else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantify SSR's within-word vs. between-word geometric effect.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("baseline_json", help="Baseline (one vector per word) JSON file.")
    parser.add_argument("ssr_json", help="SSR-treated (multiple senses per word) JSON file.")
    parser.add_argument("--exclude-defaults", action="store_true", default=True,
                        help="Exclude words whose only sense is 'default' (SSR failures).")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Show top N words by within/between ratio.")
    parser.add_argument("--csv", default=None,
                        help="If set, write full per-word table to this CSV file.")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("WITHIN- vs BETWEEN-WORD DISTANCE: BASELINE vs SSR")
    print("=" * 70)
    
    # Load
    print(f"\nLoading baseline: {args.baseline_json}")
    base_grouped, base_keys = load_with_keys(args.baseline_json)
    print(f"  {len(base_grouped)} words, {sum(v.shape[0] for v in base_grouped.values())} vectors total")
    
    print(f"Loading SSR: {args.ssr_json}")
    ssr_grouped, ssr_keys = load_with_keys(args.ssr_json)
    print(f"  {len(ssr_grouped)} words, {sum(v.shape[0] for v in ssr_grouped.values())} vectors total")
    
    if args.exclude_defaults:
        n_before = len(ssr_grouped)
        ssr_grouped = filter_defaults(ssr_grouped, ssr_keys)
        n_after = len(ssr_grouped)
        if n_before != n_after:
            print(f"\nExcluded {n_before - n_after} words whose only sense is 'default'")
    
    # Compute metrics
    print("\n" + "-" * 70)
    print("Computing per-word metrics...")
    print("-" * 70)
    
    base_rows = per_word_metrics(base_grouped)
    ssr_rows = per_word_metrics(ssr_grouped)
    
    # ----- Baseline -----
    print("\n[BASELINE]")
    base_summary = summary_stats(base_rows)
    print(f"  n_words = {base_summary['n']}")
    print(f"  d_within   = 0.000 (one vector per word, by construction)")
    print(f"  d_between  mean={base_summary['mean_d_between']:.4f}, "
          f"median={base_summary['median_d_between']:.4f}")
    
    # ----- SSR: all words -----
    print("\n[SSR, all words]")
    ssr_all = summary_stats(ssr_rows)
    print(f"  n_words = {ssr_all['n']}")
    print(f"  d_within   mean={ssr_all['mean_d_within']:.4f}, "
          f"median={ssr_all['median_d_within']:.4f}")
    print(f"  d_between  mean={ssr_all['mean_d_between']:.4f}, "
          f"median={ssr_all['median_d_between']:.4f}")
    print(f"  ratio      mean={ssr_all['mean_ratio']:.4f}, "
          f"median={ssr_all['median_ratio']:.4f}")
    
    # ----- SSR: polysemous only (n_senses >= 2) -----
    print("\n[SSR, polysemous only (n_senses >= 2)]")
    ssr_poly = summary_stats(ssr_rows, filter_polysemous=True)
    print(f"  n_words = {ssr_poly['n']}")
    print(f"  d_within   mean={ssr_poly['mean_d_within']:.4f}, "
          f"median={ssr_poly['median_d_within']:.4f}")
    print(f"  d_between  mean={ssr_poly['mean_d_between']:.4f}, "
          f"median={ssr_poly['median_d_between']:.4f}")
    print(f"  ratio      mean={ssr_poly['mean_ratio']:.4f}, "
          f"median={ssr_poly['median_ratio']:.4f}")
    
    # ----- The key comparison -----
    print("\n" + "=" * 70)
    print("KEY COMPARISON: between-word distance, baseline vs SSR")
    print("=" * 70)
    base_dbet = base_summary['mean_d_between']
    ssr_dbet = ssr_all['mean_d_between']
    delta = ssr_dbet - base_dbet
    delta_pct = 100.0 * delta / base_dbet if base_dbet > 0 else float('nan')
    print(f"  baseline d_between mean = {base_dbet:.4f}")
    print(f"  SSR      d_between mean = {ssr_dbet:.4f}")
    print(f"  delta                   = {delta:+.4f}  ({delta_pct:+.2f}%)")
    
    print(f"\n  Interpretation: SSR's d_within ({ssr_poly['mean_d_within']:.4f}) "
          f"vs SSR's d_between ({ssr_poly['mean_d_between']:.4f})")
    print(f"  ratio = {ssr_poly['mean_ratio']:.4f}")
    if ssr_poly['mean_ratio'] < 1.0:
        print(f"  -> d_within < d_between: senses of a word stay CLOSER to each other")
        print(f"     than to other words (LOCAL refinement preserves global structure).")
    else:
        print(f"  -> d_within > d_between: senses of a word are MORE SPREAD than")
        print(f"     other words around them (cross-cluster movement dominates).")
    
    # ----- Top words by ratio -----
    print("\n" + "-" * 70)
    print(f"Top {args.top_n} polysemous words by d_within / d_between ratio (largest first)")
    print("-" * 70)
    poly_rows = [r for r in ssr_rows if r['n_senses'] >= 2 and not np.isnan(r['ratio'])]
    poly_rows.sort(key=lambda r: -r['ratio'])
    print(f"{'word':<20} {'n_senses':<10} {'d_within':<12} {'d_between':<12} {'ratio':<8}")
    for r in poly_rows[:args.top_n]:
        print(f"{r['word']:<20} {r['n_senses']:<10} {r['d_within']:<12.4f} "
              f"{r['d_between']:<12.4f} {r['ratio']:<8.4f}")
    
    print(f"\nBottom {args.top_n} polysemous words by ratio (smallest first)")
    for r in poly_rows[-args.top_n:]:
        print(f"{r['word']:<20} {r['n_senses']:<10} {r['d_within']:<12.4f} "
              f"{r['d_between']:<12.4f} {r['ratio']:<8.4f}")
    
    # ----- CSV output -----
    if args.csv:
        import csv as csvmod
        print(f"\nWriting full table to {args.csv}...")
        with open(args.csv, 'w', newline='') as f:
            w = csvmod.DictWriter(f, fieldnames=['word', 'n_senses', 'd_within', 'd_between', 'ratio'])
            w.writeheader()
            for r in sorted(ssr_rows, key=lambda r: -r.get('ratio', 0) if not np.isnan(r.get('ratio', float('nan'))) else 0):
                w.writerow(r)
        print("Done.")


if __name__ == "__main__":
    main()
