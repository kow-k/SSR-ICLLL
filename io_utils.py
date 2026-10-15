#!/usr/bin/env python3
"""
io_utils.py — minimal embedding loader for the SSR validation scripts.

Loads a plain-text or gzipped embedding file (bare GloVe, or with a gensim
"<n> <dim>" header), keeps rank order (= frequency), unit-normalizes rows, and
returns an ``Emb`` object with a cosine nearest-neighbor helper. Self-contained;
depends only on numpy.
"""

import numpy as np

RANK_CAP = 40000


class Emb:
    def __init__(self, words, matrix):
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        self.M = matrix
        self.vocab = set(words)

    def neighbors(self, word, top_k):
        i = self.index[word]
        sims = self.M @ self.M[i]
        sims[i] = -np.inf
        idx = np.argpartition(-sims, top_k)[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [self.words[j] for j in idx]


def _unit_rows(A):
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-10
    return A / n


def load_glove(path, max_words=RANK_CAP):
    """gzip/plain, bare GloVe or gensim header. Keeps rank order (= frequency)."""
    import gzip
    with open(path, "rb") as raw:
        magic = raw.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    words, rows = [], []
    with opener(path, "rt", encoding="utf8", errors="replace") as f:
        first = f.readline().rstrip().split(" ")
        if not (len(first) == 2 and all(t.isdigit() for t in first)):
            f.seek(0)
        for line in f:
            p = line.rstrip().split(" ")
            if len(p) < 3:
                continue
            words.append(p[0])
            rows.append(np.asarray(p[1:], dtype=np.float32))
            if len(words) >= max_words:
                break
    return Emb(words, _unit_rows(np.vstack(rows)))
