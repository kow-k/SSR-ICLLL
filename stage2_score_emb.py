#!/usr/bin/env python3
"""stage2_score_emb.py -- embedding-similarity scorer for Stage 2. Replaces
sparse word-overlap with centroid cosine between each SSR cluster and each
synset's member set (members embedded via the same space), the way
synset_mapper bridges the vocabulary gap. Assignment via Hungarian."""
import numpy as np
from stage1_anchor import members
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def syn_centroid(mem, emb):
    vs = [emb.M[emb.index[w]] for w in mem if w in emb.index]
    return _unit(np.mean(vs, axis=0)) if vs else None

def cl_centroid(cluster, emb):
    vs = [emb.M[emb.index[w]] for w in cluster if w in emb.index]
    return _unit(np.mean(vs, axis=0)) if vs else None

def assign_and_score_emb(clusters, S, M, emb, member_cache):
    """Map clusters<->synsets by centroid cosine (Hungarian). M recovered iff
    the cluster assigned to M has M as its argmax synset."""
    Mj = [j for j, s in enumerate(S) if s.name() == M.name()][0]
    syn_c = [syn_centroid(member_cache[s.name()], emb) for s in S]
    if syn_c[Mj] is None:
        return False, "M-no-vocab"
    cl_c = [cl_centroid(c, emb) for c in clusters]
    C, K = len(clusters), len(S)
    O = np.full((C, K), -1.0)
    for i in range(C):
        if cl_c[i] is None:
            continue
        for j in range(K):
            if syn_c[j] is not None:
                O[i, j] = float(cl_c[i] @ syn_c[j])
    if HAVE_SCIPY:
        ri, ci = linear_sum_assignment(-O)
        amap = {int(r): int(c) for r, c in zip(ri, ci)}
    else:
        amap = {}; used = set()
        for i in np.argsort(-O.max(axis=1)):
            for j in np.argsort(-O[i]):
                if j not in used:
                    amap[i] = int(j); used.add(int(j)); break
    cl_for_M = [i for i, j in amap.items() if j == Mj]
    if not cl_for_M:
        return False, "M-unassigned"
    i = cl_for_M[0]
    if int(np.argmax(O[i])) != Mj:
        return False, "sibling-wins"
    return True, "recovered"