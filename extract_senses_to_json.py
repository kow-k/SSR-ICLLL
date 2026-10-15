#!/usr/bin/env python3
"""
Extract sense vectors for words listed in polysemy_words_300.csv using SSR,
and save the results as a JSON file suitable for sense_cluster_radial.py.

Usage:
    python extract_senses_to_json.py <embedding_path> <words_csv> <output_json> [mode]

    mode: 'discover' (unsupervised, default) or 'induce' (anchor-guided)

Examples:
    # Unsupervised (Paper 1's method)
    python extract_senses_to_json.py glove.6B.100d.txt polysemy_words_300.csv senses_100d.json discover
    
    # Anchor-guided (Paper 2's method, uses hybrid FrameNet+WordNet anchors)
    python extract_senses_to_json.py glove.6B.100d.txt polysemy_words_300.csv senses_100d.json induce
"""

import csv
import json
import sys
import numpy as np
from sense_explorer import SenseExplorer


def load_word_list(csv_path: str) -> list:
    """
    Load words from CSV. Returns list of dicts with word and expected_senses.
    
    Supports two CSV formats:
    - Bare-word CSV (e.g., polysemy_words_300.csv): uses 'word' column for both
      display and vocabulary lookup.
    - POS-tagged CSV (e.g., bnc_verbs_120.csv): uses 'word' column for display
      and 'bnc_token' column (e.g., 'run_VERB') for vocabulary lookup.
    """
    words = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Accept either 'word' or 'verb' as the display column
            display_word = (row.get('word') or row.get('verb') or '').strip()
            if not display_word or display_word.startswith('#'):
                continue
            
            # Use bnc_token for lookup if present, else fall back to display word
            lookup_token = row.get('bnc_token', '').strip() or display_word
            
            words.append({
                'word': display_word,
                'lookup_token': lookup_token,
                'expected_senses': int(row['expected_senses']),
                'domain': (row.get('domain_primary') or row.get('semantic_class') or '').strip(),
            })
    return words


def extract_senses(explorer, word: str, n_senses: int, mode: str) -> dict:
    """
    Run SSR on a single word.
    
    Args:
        explorer: SenseExplorer instance
        word: Target word
        n_senses: Expected number of senses
        mode: 'discover' (unsupervised) or 'induce' (anchor-guided)
    
    Returns:
        Dict mapping sense_name -> vector (list), or None on failure
    """
    try:
        if mode == 'discover':
            # Unsupervised: spectral clustering on neighborhood + self-repair
            senses = explorer.discover_senses(
                word=word,
                n_senses=n_senses,
                clustering_method='spectral'
            )
        elif mode == 'induce':
            # Anchor-guided: uses hybrid FrameNet/WordNet anchors automatically
            senses = explorer.induce_senses(
                word=word,
                n_senses=n_senses
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Convert numpy arrays to lists for JSON serialization
        return {sense_name: vector.tolist() for sense_name, vector in senses.items()}
    
    except Exception as e:
        print(f"  WARNING: Could not extract senses for '{word}': {e}")
        return None


def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_senses_to_json.py <embedding_path> <words_csv> <output_json> [mode]")
        print("  mode: 'discover' (unsupervised, default) or 'induce' (anchor-guided)")
        sys.exit(1)
    
    embedding_path = sys.argv[1]
    words_csv = sys.argv[2]
    output_json = sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else 'discover'
    
    if mode not in ('discover', 'induce'):
        print(f"ERROR: mode must be 'discover' or 'induce', got '{mode}'")
        sys.exit(1)
    
    print(f"Mode: {mode} ({'unsupervised' if mode == 'discover' else 'anchor-guided'})")
    
    # Load words first
    words = load_word_list(words_csv)
    print(f"Loaded {len(words)} words from {words_csv}")
    
    # Initialize SenseExplorer from GloVe file (handles loading internally)
    print(f"\nInitializing SenseExplorer from {embedding_path}...")
    explorer = SenseExplorer.from_glove(embedding_path, verbose=False)
    
    # Process each word
    results = {}
    skipped = []
    
    for i, w in enumerate(words):
        word = w['word']                # display name (e.g., 'walk')
        lookup = w['lookup_token']      # vocabulary key (e.g., 'walk_VERB')
        expected = w['expected_senses']
        
        print(f"[{i+1}/{len(words)}] Processing '{lookup}' (expecting {expected} senses)...")
        
        if lookup not in explorer.vocab:
            print(f"  SKIPPED: '{lookup}' not in vocabulary")
            skipped.append(lookup)
            continue
        
        sense_vectors = extract_senses(explorer, lookup, expected, mode)
        
        if sense_vectors:
            # Handle duplicate display words in CSV by suffixing with domain
            if word in results:
                domain = w.get('domain', f'dup{i}')
                key = f"{word}_{domain}"
            else:
                key = word
            
            results[key] = sense_vectors
            print(f"  OK: extracted {len(sense_vectors)} senses ({', '.join(sense_vectors.keys())})")
        else:
            skipped.append(lookup)
    
    # Save to JSON
    print(f"\nSaving {len(results)} words to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone!")
    print(f"  Successfully processed: {len(results)}")
    print(f"  Skipped: {len(skipped)}")
    if skipped:
        print(f"  Skipped words: {', '.join(skipped[:20])}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped)-20} more")


if __name__ == "__main__":
    main()
