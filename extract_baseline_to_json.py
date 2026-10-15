#!/usr/bin/env python3
"""
Extract baseline embeddings (no sense separation) for a list of words,
in the same JSON format as extract_senses_to_json.py output.

This is the CONTROL condition for the cross-word clustering experiment:
each word gets one vector (its bare embedding), labeled as 'baseline'.
The output can be fed to sense_cluster_radial.py for comparison against
the SSR-treated version.

Usage:
    python extract_baseline_to_json.py <embedding_path> <words_csv> <output_json>

Example:
    python extract_baseline_to_json.py bnc-w2v-model.txt bnc_verbs_120.csv bnc_verbs_120_baseline.json
"""

import csv
import json
import sys
from sense_explorer import SenseExplorer


def load_word_list(csv_path: str) -> list:
    """Load words from CSV. Accepts 'word' or 'verb' as display column,
    'bnc_token' as optional lookup column."""
    words = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            display_word = (row.get('word') or row.get('verb') or '').strip()
            if not display_word or display_word.startswith('#'):
                continue
            lookup_token = row.get('bnc_token', '').strip() or display_word
            words.append({
                'word': display_word,
                'lookup_token': lookup_token,
                'domain': (row.get('domain_primary') or row.get('semantic_class') or '').strip(),
            })
    return words


def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_baseline_to_json.py <embedding_path> <words_csv> <output_json>")
        sys.exit(1)
    
    embedding_path = sys.argv[1]
    words_csv = sys.argv[2]
    output_json = sys.argv[3]
    
    words = load_word_list(words_csv)
    print(f"Loaded {len(words)} words from {words_csv}")
    
    print(f"\nInitializing SenseExplorer from {embedding_path}...")
    explorer = SenseExplorer.from_glove(embedding_path, verbose=False)
    
    results = {}
    skipped = []
    
    for i, w in enumerate(words):
        word = w['word']
        lookup = w['lookup_token']
        
        if lookup not in explorer.vocab:
            print(f"[{i+1}/{len(words)}] SKIPPED: '{lookup}' not in vocabulary")
            skipped.append(lookup)
            continue
        
        # Get the bare embedding (no SSR)
        vec = explorer.embeddings[lookup]
        
        # Use bare word as key, suffix with domain if duplicate
        if word in results:
            domain = w.get('domain', f'dup{i}')
            key = f"{word}_{domain}"
        else:
            key = word
        
        # Same JSON schema as SSR output: {word: {sense_name: vector}}
        results[key] = {'baseline': vec.tolist()}
        print(f"[{i+1}/{len(words)}] OK: '{lookup}'")
    
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
