#!/usr/bin/env python3
"""
centroids_only.py -- run any SSR driver script with the repair step removed.

Spectral Sense Recovery = steps (1)+(2) of the original pipeline:
  K nearest neighbors -> spectral clustering into k groups -> one centroid per group.
This wrapper replaces SenseExplorer._simulated_repair with the identity on the
centroids, so every method (discover_senses, discover_senses_auto,
induce_senses, ...) returns the cluster centroids instead of repaired vectors.
Nothing else changes: same neighbors, same k, same clustering, same scoring.

Usage (from the repository root):
    python centroids_only.py <driver.py> [driver args ...]
    python centroids_only.py --seed 0 run_synset_mapping.py --glove ... --preset homonym

The driver runs exactly as if invoked directly (sys.argv and __name__ are set).
"""
import sys, os, runpy
import numpy as np


def patch():
    """Make every SenseExplorer method return its normalized cluster centroids."""
    from sense_explorer import core

    def _centroids(self, word, sense_centroids, noise_level, sense_loyal=True):
        return {s: np.asarray(c) / (np.linalg.norm(c) + 1e-10)
                for s, c in sense_centroids.items()}

    core.SenseExplorer._simulated_repair = _centroids
    return core.SenseExplorer


def main():
    args = sys.argv[1:]
    seed = 0
    if args[:1] == ["--seed"]:
        seed, args = int(args[1]), args[2:]
    if not args or not args[0].endswith(".py"):
        sys.exit(__doc__)
    sys.path.insert(0, os.getcwd())
    np.random.seed(seed)
    patch()
    print(f"[centroids_only] repair disabled: senses = spectral cluster centroids "
          f"(seed={seed})", file=sys.stderr)
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
