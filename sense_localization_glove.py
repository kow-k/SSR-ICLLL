#!/usr/bin/env python3
"""
Sense Vector Localization in GloVe Embeddings
===============================================
Given sense vectors extracted by SSR (via SenseExplorer),
decompose the word vector to reveal WHERE each sense lives
and analyze the geometric relationships between senses.

Usage:
    # Option A: With SenseExplorer installed
    python sense_localization_glove.py --glove glove.6B.100d.txt
    
    # Option B: Specify custom GloVe path
    python sense_localization_glove.py --glove /path/to/glove.6B.100d.txt
    
    # Option C: Also test with 300d
    python sense_localization_glove.py --glove glove.6B.300d.txt

Requirements:
    pip install numpy matplotlib scikit-learn

Author: Kow Kuroda & Claude
Date: February 2026
"""

import numpy as np
from numpy.linalg import norm, lstsq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import argparse
import os
import time
import sys


# =============================================================================
# Embedding Loader
# =============================================================================

def load_glove(filepath: str, max_words: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Load GloVe embeddings from text file."""
    embeddings = {}
    dim = None
    
    print(f"Loading embeddings from {filepath}...", flush=True)
    start = time.time()
    
    count = 0
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                embeddings[word] = vec
                if dim is None:
                    dim = len(vec)
                count += 1
                if max_words and count >= max_words:
                    break
            except ValueError:
                continue
    
    elapsed = time.time() - start
    print(f"Loaded {len(embeddings):,} embeddings, dim={dim} ({elapsed:.1f}s)")
    return embeddings


def normalize_embeddings(embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """L2-normalize all embeddings."""
    return {w: v / (norm(v) + 1e-10) for w, v in embeddings.items()}


# =============================================================================
# SSR (Simulated Self-Repair) — Standalone Implementation
# =============================================================================
# This is a self-contained SSR so the script works without SenseExplorer.
# If SenseExplorer is available, it will be used instead (see main()).

# Anchor word sets for polysemous words
SENSE_ANCHORS = {
    'bank': {
        'financial': ['money', 'loan', 'deposit', 'account', 'finance', 'credit',
                       'cash', 'investment', 'interest', 'debt'],
        'river': ['river', 'shore', 'water', 'stream', 'flood', 'creek',
                  'lake', 'erosion', 'sand', 'valley']
    },
    'bat': {
        'animal': ['wing', 'cave', 'fly', 'nocturnal', 'mammal', 'vampire',
                   'sonar', 'insect', 'roost', 'echolocation'],
        'sports': ['ball', 'hit', 'swing', 'baseball', 'game', 'player',
                   'pitch', 'home', 'innings', 'cricket']
    },
    'cell': {
        'biology': ['organism', 'membrane', 'nucleus', 'tissue', 'blood',
                    'bacteria', 'protein', 'division', 'mitosis', 'dna'],
        'prison': ['jail', 'prisoner', 'guard', 'bars', 'inmate', 'locked',
                   'confined', 'sentence', 'detention', 'solitary'],
        'phone': ['phone', 'mobile', 'call', 'wireless', 'signal', 'battery',
                  'network', 'device', 'cellular', 'smartphone']
    },
    'crane': {
        'bird': ['bird', 'fly', 'wing', 'migrate', 'nest', 'feather',
                 'beak', 'flock', 'heron', 'stork'],
        'machine': ['lift', 'construction', 'heavy', 'steel', 'tower',
                    'cable', 'load', 'build', 'hoist', 'derrick']
    },
    'mouse': {
        'animal': ['cheese', 'cat', 'trap', 'rodent', 'tail', 'squeak',
                   'hole', 'pest', 'rat', 'mice'],
        'computer': ['click', 'cursor', 'keyboard', 'screen', 'button',
                     'pointer', 'scroll', 'pad', 'trackball', 'usb']
    },
    'plant': {
        'botanical': ['leaf', 'grow', 'flower', 'root', 'seed', 'green',
                      'garden', 'soil', 'photosynthesis', 'tree'],
        'industrial': ['factory', 'manufacture', 'worker', 'production',
                       'facility', 'power', 'nuclear', 'chemical', 'refinery', 'mill']
    },
    'spring': {
        'season': ['summer', 'winter', 'autumn', 'bloom', 'warm', 'flowers',
                   'april', 'march', 'seasonal', 'equinox'],
        'water': ['fountain', 'well', 'source', 'fresh', 'mineral', 'natural',
                  'hot', 'geyser', 'underground', 'bubbling'],
        'coil': ['coil', 'bounce', 'metal', 'tension', 'elastic', 'mattress',
                 'shock', 'pressure', 'suspension', 'loaded']
    },
    'rock': {
        'stone': ['stone', 'mineral', 'boulder', 'cliff', 'mountain', 'geology',
                  'hard', 'solid', 'granite', 'sediment'],
        'music': ['band', 'guitar', 'concert', 'song', 'roll', 'punk',
                  'metal', 'pop', 'drums', 'album']
    },
    'star': {
        'astronomy': ['sky', 'sun', 'planet', 'galaxy', 'space', 'light',
                      'bright', 'constellation', 'solar', 'nebula'],
        'celebrity': ['movie', 'famous', 'actor', 'hollywood', 'celebrity',
                      'fame', 'talent', 'show', 'award', 'premiere']
    },
    'bark': {
        'tree': ['tree', 'wood', 'trunk', 'forest', 'oak', 'pine', 'branch',
                 'layer', 'cork', 'birch'],
        'dog': ['dog', 'sound', 'loud', 'growl', 'pet', 'animal', 'noise',
                'woof', 'howl', 'puppy']
    },
    # --- Highly polysemous verbs ---
    'run': {
        'locomotion': ['sprint', 'jog', 'dash', 'race', 'marathon', 'track',
                       'athlete', 'faster', 'mile', 'stride'],
        'operate': ['manage', 'operate', 'business', 'company', 'director',
                    'organization', 'administration', 'lead', 'oversee', 'charge'],
        'flow': ['flow', 'drip', 'faucet', 'stream', 'water', 'leak',
                 'pour', 'tap', 'pipe', 'spill'],
        'computing': ['execute', 'program', 'software', 'code', 'process',
                      'script', 'command', 'debug', 'compile', 'server'],
    },
    'take': {
        'seize': ['grab', 'seize', 'grasp', 'hold', 'snatch', 'pick',
                  'remove', 'steal', 'capture', 'clutch'],
        'transport': ['bring', 'carry', 'transport', 'deliver', 'move',
                      'drive', 'ride', 'bus', 'taxi', 'passenger'],
        'consume': ['eat', 'drink', 'swallow', 'medicine', 'pill', 'dose',
                    'ingest', 'tablet', 'prescription', 'vitamin'],
        'require': ['need', 'require', 'demand', 'cost', 'effort', 'time',
                    'patience', 'skill', 'hour', 'minute'],
    },
    'set': {
        'place': ['put', 'place', 'lay', 'arrange', 'position', 'table',
                  'shelf', 'down', 'aside', 'prepare'],
        'collection': ['group', 'collection', 'series', 'kit', 'pair',
                       'assortment', 'batch', 'bundle', 'pack', 'array'],
        'configure': ['adjust', 'configure', 'setting', 'parameter', 'dial',
                      'thermostat', 'clock', 'timer', 'mode', 'default'],
    },
    'break': {
        'fracture': ['crack', 'shatter', 'smash', 'fracture', 'split',
                     'snap', 'crush', 'broken', 'fragment', 'piece'],
        'pause': ['rest', 'pause', 'stop', 'intermission', 'recess', 'lunch',
                  'holiday', 'vacation', 'timeout', 'breather'],
        'violate': ['violate', 'breach', 'disobey', 'rule', 'law',
                    'promise', 'contract', 'regulation', 'violation', 'offense'],
    },
    'light': {
        'illumination': ['bright', 'lamp', 'sun', 'glow', 'shine', 'beam',
                         'ray', 'bulb', 'lantern', 'candle'],
        'weight': ['lightweight', 'feather', 'thin', 'delicate', 'airy',
                   'portable', 'weightless', 'floating', 'gentle', 'soft'],
        'ignite': ['fire', 'match', 'flame', 'burn', 'ignite', 'spark',
                   'cigarette', 'torch', 'kindle', 'lighter'],
    },
    'match': {
        'contest': ['game', 'tournament', 'opponent', 'championship', 'team',
                    'competition', 'soccer', 'tennis', 'wrestling', 'boxing'],
        'correspond': ['fit', 'correspond', 'pair', 'equal', 'identical',
                       'compatible', 'similar', 'complement', 'align', 'agree'],
        'fire_stick': ['fire', 'flame', 'strike', 'sulfur', 'matchbox',
                       'ignite', 'burn', 'stick', 'phosphorus', 'wick'],
    },
    'board': {
        'plank': ['wood', 'plank', 'panel', 'timber', 'lumber', 'plywood',
                  'floor', 'shelf', 'wooden', 'carpenter'],
        'committee': ['director', 'committee', 'member', 'chairman', 'meeting',
                      'trustee', 'governance', 'executive', 'council', 'advisory'],
        'embark': ['plane', 'ship', 'train', 'passenger', 'gate', 'ticket',
                   'flight', 'departure', 'airport', 'embark'],
    },
    # --- Additional polysemous nouns ---
    'book': {
        'text': ['read', 'novel', 'author', 'chapter', 'page', 'library',
                 'publish', 'story', 'fiction', 'literature'],
        'reserve': ['reservation', 'hotel', 'ticket', 'appointment', 'schedule',
                    'reserve', 'booking', 'confirm', 'arrange', 'advance'],
    },
    'bass': {
        'fish': ['fish', 'fishing', 'lake', 'trout', 'catch', 'angler',
                 'freshwater', 'bait', 'rod', 'salmon'],
        'music': ['guitar', 'sound', 'tone', 'deep', 'frequency', 'speaker',
                  'drum', 'rhythm', 'amplifier', 'jazz'],
    },
    'palm': {
        'hand': ['hand', 'finger', 'grip', 'touch', 'wrist', 'fist',
                 'thumb', 'skin', 'slap', 'gesture'],
        'tree': ['tree', 'tropical', 'coconut', 'beach', 'frond', 'island',
                 'shade', 'leaf', 'grove', 'plantation'],
    },
    'suit': {
        'clothing': ['jacket', 'tie', 'shirt', 'wear', 'formal', 'tailor',
                     'pants', 'dress', 'fashion', 'outfit'],
        'legal': ['lawsuit', 'court', 'judge', 'lawyer', 'plaintiff', 'filed',
                  'trial', 'damages', 'attorney', 'litigation'],
        'cards': ['spade', 'heart', 'diamond', 'club', 'card', 'deck',
                  'trump', 'ace', 'poker', 'dealt'],
    },
    'pitch': {
        'throw': ['throw', 'ball', 'baseball', 'pitcher', 'mound', 'fastball',
                  'curveball', 'strike', 'innings', 'catcher'],
        'sound': ['tone', 'frequency', 'sound', 'note', 'high', 'low',
                  'melody', 'tune', 'voice', 'octave'],
        'sales': ['proposal', 'presentation', 'investor', 'sell', 'persuade',
                  'idea', 'startup', 'venture', 'elevator', 'convince'],
    },
    'draft': {
        'writing': ['manuscript', 'revision', 'edit', 'write', 'essay',
                    'document', 'rough', 'outline', 'version', 'proofread'],
        'military': ['conscription', 'enlist', 'army', 'soldier', 'recruit',
                     'service', 'war', 'military', 'selective', 'duty'],
        'beer': ['beer', 'tap', 'brew', 'ale', 'lager', 'pint',
                 'keg', 'bartender', 'pub', 'stout'],
    },
    'note': {
        'writing': ['write', 'memo', 'message', 'letter', 'jot', 'scribble',
                    'pad', 'paper', 'reminder', 'sticky'],
        'music': ['melody', 'tone', 'chord', 'scale', 'octave', 'sharp',
                  'flat', 'pitch', 'rhythm', 'harmony'],
        'money': ['bill', 'currency', 'banknote', 'dollar', 'denomination',
                  'cash', 'treasury', 'mint', 'tender', 'pound'],
    },
    'port': {
        'harbor': ['harbor', 'ship', 'dock', 'cargo', 'maritime', 'vessel',
                   'wharf', 'pier', 'berth', 'shipping'],
        'computing': ['socket', 'connection', 'network', 'usb', 'serial',
                      'ethernet', 'interface', 'firewall', 'tcp', 'protocol'],
        'wine': ['wine', 'portuguese', 'vintage', 'fortified', 'cellar',
                 'bottle', 'sweet', 'tawny', 'ruby', 'grape'],
    },
    'seal': {
        'animal': ['marine', 'ocean', 'arctic', 'fur', 'flipper', 'walrus',
                   'swim', 'colony', 'pup', 'beach'],
        'closure': ['stamp', 'envelope', 'wax', 'airtight', 'waterproof',
                    'gasket', 'container', 'lid', 'tight', 'secure'],
        'approval': ['official', 'approval', 'certificate', 'emblem', 'insignia',
                     'authorize', 'validate', 'notary', 'government', 'royal'],
    },
    'line': {
        'mark': ['draw', 'straight', 'curve', 'pencil', 'ruler', 'stroke',
                 'boundary', 'edge', 'border', 'segment'],
        'queue': ['wait', 'queue', 'stand', 'customer', 'front', 'long',
                  'behind', 'ahead', 'patience', 'crowded'],
        'text': ['sentence', 'paragraph', 'verse', 'poem', 'script', 'dialogue',
                 'actor', 'lyric', 'passage', 'page'],
    },
}

EXPECTED_SENSES = {
    'bank': 2, 'bat': 2, 'cell': 3, 'crane': 2, 'mouse': 2,
    'plant': 2, 'spring': 3, 'rock': 2, 'star': 2, 'bark': 2,
    'run': 4, 'take': 4, 'set': 3, 'break': 3, 'light': 3,
    'match': 3, 'board': 3,
    'book': 2, 'bass': 2, 'palm': 2, 'suit': 3, 'pitch': 3,
    'draft': 3, 'note': 3, 'port': 3, 'seal': 3, 'line': 3,
}


def run_ssr(word: str, embeddings: Dict[str, np.ndarray],
            sense_anchors: Dict[str, List[str]],
            n_copies: int = 50, noise_level: float = 0.3,
            seed_strength: float = 0.5, n_iterations: int = 30,
            anchor_pull: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Simulated Self-Repair (sense-loyal mode) for sense vector extraction.
    
    Standalone implementation matching SenseExplorer.induce_senses() behavior.
    
    Args:
        word: Target polysemous word
        embeddings: Full embedding dictionary
        sense_anchors: {sense_label: [anchor_words]}
        n_copies: Number of noisy copies per sense
        noise_level: Perturbation magnitude
        seed_strength: How strongly copies are seeded toward sense centroids
        n_iterations: Self-repair iterations
        anchor_pull: How strongly anchor centroids pull during repair
        
    Returns:
        {sense_label: sense_vector} extracted sense vectors
    """
    if word not in embeddings:
        print(f"  WARNING: '{word}' not in embeddings, skipping")
        return {}
    
    w = embeddings[word]
    d = len(w)
    
    # Compute anchor centroids (only from words present in embeddings)
    anchor_centroids = {}
    for label, anchors in sense_anchors.items():
        vecs = [embeddings[a] for a in anchors if a in embeddings]
        if vecs:
            centroid = np.mean(vecs, axis=0)
            anchor_centroids[label] = centroid / (norm(centroid) + 1e-10)
        else:
            print(f"  WARNING: No anchor words found for sense '{label}' of '{word}'")
    
    if len(anchor_centroids) < 2:
        print(f"  WARNING: Less than 2 senses have valid anchors for '{word}'")
        return {}
    
    senses = list(anchor_centroids.keys())
    copies_per_sense = n_copies // len(senses)
    
    # --- 1. Create seeded noisy copies ---
    all_copies = []
    copy_assignments = []  # sense-loyal: each copy remembers its sense
    
    for sense_idx, sense in enumerate(senses):
        centroid = anchor_centroids[sense]
        for _ in range(copies_per_sense):
            copy = w.copy()
            
            # Add noise: perturb random subset of dimensions
            n_perturb = np.random.randint(int(d * 0.5), int(d * 0.8))
            perturb_dims = np.random.choice(d, n_perturb, replace=False)
            for dim_i in perturb_dims:
                copy[dim_i] += np.random.randn() * abs(w[dim_i]) * noise_level
            
            # Seed toward sense centroid
            copy_normed = copy / (norm(copy) + 1e-10)
            direction = centroid - copy_normed
            copy += seed_strength * direction * norm(copy)
            
            all_copies.append(copy)
            copy_assignments.append(sense)
    
    copies = np.array(all_copies)
    
    # --- 2. Sense-loyal self-repair ---
    for iteration in range(n_iterations):
        # Compute current cluster centroids
        cluster_centroids = {}
        for sense in senses:
            members = [copies[i] for i in range(len(copies)) 
                       if copy_assignments[i] == sense]
            if members:
                cluster_centroids[sense] = np.mean(members, axis=0)
            else:
                cluster_centroids[sense] = anchor_centroids[sense]
        
        # Pull each copy toward its assigned sense centroid + anchor centroid
        repair_rate = 0.1
        for i in range(len(copies)):
            sense = copy_assignments[i]
            # Pull toward cluster centroid
            copies[i] += repair_rate * (cluster_centroids[sense] - copies[i])
            # Anchor pull
            copies[i] += anchor_pull * (anchor_centroids[sense] - copies[i])
    
    # --- 3. Extract sense vectors (cluster centroids) ---
    sense_vectors = {}
    for sense in senses:
        members = [copies[i] for i in range(len(copies))
                   if copy_assignments[i] == sense]
        if members:
            vec = np.mean(members, axis=0)
            sense_vectors[sense] = vec
    
    return sense_vectors


# =============================================================================
# Sense Decomposition
# =============================================================================

@dataclass
class SenseDecomposition:
    """Complete result of decomposing a word vector into sense components."""
    word: str
    word_vector: np.ndarray
    sense_labels: List[str]
    sense_vectors: np.ndarray
    coefficients: np.ndarray
    sense_contributions: np.ndarray
    residual: np.ndarray
    reconstruction: np.ndarray
    variance_explained_total: float
    variance_explained_per_sense: np.ndarray
    residual_norm_ratio: float
    dimension_attribution: np.ndarray
    dimension_dominance: np.ndarray
    shared_dimensions: np.ndarray
    unique_dimensions: Dict[str, np.ndarray]
    contested_dimensions: np.ndarray
    sense_cosine_matrix: np.ndarray
    sense_angles_deg: np.ndarray
    word_sense_cosines: np.ndarray
    word_sense_projections: np.ndarray
    sign_agreement: np.ndarray
    constructive_dims: Dict[str, np.ndarray]
    destructive_dims: Dict[str, np.ndarray]


def decompose(word: str, word_vector: np.ndarray,
              sense_dict: Dict[str, np.ndarray],
              sharing_threshold: float = 0.55) -> SenseDecomposition:
    """
    Full decomposition of word vector into sense components.
    
    w = α₁s₁ + α₂s₂ + ... + αₖsₖ + r
    """
    w = word_vector.copy().astype(float)
    d = len(w)
    
    sense_labels = list(sense_dict.keys())
    k = len(sense_labels)
    S = np.array([sense_dict[label] for label in sense_labels], dtype=float)
    
    # 1. Linear decomposition
    alpha, _, _, _ = lstsq(S.T, w, rcond=None)
    reconstruction = S.T @ alpha
    residual = w - reconstruction
    sense_contributions = alpha[:, np.newaxis] * S
    
    # 2. Variance analysis
    w_var = np.sum(w ** 2)
    r_var = np.sum(residual ** 2)
    var_explained_total = 1.0 - r_var / w_var if w_var > 0 else 0.0
    residual_norm_ratio = norm(residual) / norm(w) if norm(w) > 0 else 0.0
    var_per_sense = np.array([
        np.sum(sense_contributions[i] ** 2) / w_var if w_var > 0 else 0.0
        for i in range(k)
    ])
    
    # 3. Dimensional analysis
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
    
    # 4. Sense geometry
    cosine_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ni, nj = norm(S[i]), norm(S[j])
            cosine_matrix[i, j] = np.dot(S[i], S[j]) / (ni * nj) if ni > 0 and nj > 0 else 0.0
    
    angles_deg = np.degrees(np.arccos(np.clip(cosine_matrix, -1, 1)))
    
    w_norm = norm(w)
    word_sense_cosines = np.array([
        np.dot(w, S[i]) / (w_norm * norm(S[i])) if w_norm > 0 and norm(S[i]) > 0 else 0.0
        for i in range(k)
    ])
    word_sense_projections = np.array([
        np.dot(w, S[i]) / norm(S[i]) if norm(S[i]) > 0 else 0.0
        for i in range(k)
    ])
    
    # 5. Sign analysis
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
# Text Reports
# =============================================================================

def print_report(d: SenseDecomposition):
    """Print a text summary of the decomposition."""
    k = len(d.sense_labels)
    print(f"\n{'='*70}")
    print(f"SENSE LOCALIZATION: '{d.word}'")
    print(f"{'='*70}")
    
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
    
    if k >= 2:
        for i in range(k):
            for j in range(i+1, k):
                print(f"    ∠({d.sense_labels[i]}, {d.sense_labels[j]}) = "
                      f"{d.sense_angles_deg[i,j]:.1f}°  "
                      f"(cos = {d.sense_cosine_matrix[i,j]:.4f})")
    
    ndims = len(d.word_vector)
    n_shared = int(np.sum(d.shared_dimensions))
    n_contested = int(np.sum(d.contested_dimensions))
    print(f"\n  Dimensional Territories ({ndims} dims):")
    print(f"    Shared (no clear owner): {n_shared} ({100*n_shared/ndims:.0f}%)")
    print(f"    Contested (opposite signs): {n_contested} ({100*n_contested/ndims:.0f}%)")
    for label in d.sense_labels:
        n_unique = len(d.unique_dimensions[label])
        n_cons = len(d.constructive_dims[label])
        n_dest = len(d.destructive_dims[label])
        print(f"    {label}: {n_unique} unique dims | "
              f"{n_cons} constructive | {n_dest} destructive")


def print_cross_word_summary(decompositions: List[SenseDecomposition]):
    """Print generalizations across multiple polysemous words."""
    print(f"\n{'='*70}")
    print(f"CROSS-WORD GENERALIZATIONS ({len(decompositions)} words)")
    print(f"{'='*70}")
    
    r2_vals = [d.variance_explained_total for d in decompositions]
    resid_vals = [d.residual_norm_ratio for d in decompositions]
    
    print(f"\n  Variance Explained:")
    print(f"    Mean R²: {np.mean(r2_vals):.4f} ± {np.std(r2_vals):.4f}")
    print(f"    Range: [{min(r2_vals):.4f}, {max(r2_vals):.4f}]")
    print(f"    Mean residual ratio: {np.mean(resid_vals):.4f}")
    
    all_angles = []
    for d in decompositions:
        k = len(d.sense_labels)
        for i in range(k):
            for j in range(i+1, k):
                all_angles.append((d.word, d.sense_labels[i], d.sense_labels[j],
                                   d.sense_angles_deg[i, j]))
    
    if all_angles:
        angles_only = [a[3] for a in all_angles]
        print(f"\n  Inter-Sense Angles (THE KEY RESULT):")
        print(f"    Mean: {np.mean(angles_only):.1f}° ± {np.std(angles_only):.1f}°")
        print(f"    Range: [{min(angles_only):.1f}°, {max(angles_only):.1f}°]")
        near_orth = sum(1 for a in angles_only if 70 <= a <= 110)
        print(f"    Near-orthogonal (70-110°): {near_orth}/{len(angles_only)}")
        print(f"\n    Per word:")
        for word, s1, s2, angle in all_angles:
            orth_mark = "✓" if 70 <= angle <= 110 else "✗"
            print(f"      {word}: ∠({s1}, {s2}) = {angle:.1f}° {orth_mark}")
    
    sharing_rates = [np.sum(d.shared_dimensions) / len(d.word_vector) for d in decompositions]
    contested_rates = [np.sum(d.contested_dimensions) / len(d.word_vector) for d in decompositions]
    
    print(f"\n  Dimensional Structure:")
    print(f"    Mean sharing rate: {np.mean(sharing_rates):.2%}")
    print(f"    Mean contested rate: {np.mean(contested_rates):.2%}")
    
    print(f"\n  Coefficient Patterns (sense mixing weights):")
    for d in decompositions:
        coefs = " | ".join([f"{d.sense_labels[i]}={d.coefficients[i]:.3f}"
                            for i in range(len(d.sense_labels))])
        print(f"    {d.word}: {coefs}")
    
    print(f"\n  Dominant Sense Bias:")
    for d in decompositions:
        abs_coefs = np.abs(d.coefficients)
        max_i = np.argmax(abs_coefs)
        min_i = np.argmin(abs_coefs)
        ratio = abs_coefs[max_i] / (abs_coefs[min_i] + 1e-10)
        print(f"    {d.word}: dominant = {d.sense_labels[max_i]} "
              f"({ratio:.2f}x over {d.sense_labels[min_i]})")


# =============================================================================
# Visualization
# =============================================================================

SENSE_COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
                '#00BCD4', '#795548', '#607D8B']
RESIDUAL_COLOR = '#BDBDBD'


def plot_dimension_attribution(decomp, ax, sort_by_dominance=False, title=None):
    d = len(decomp.word_vector)
    k = len(decomp.sense_labels)
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
    d = len(decomp.word_vector)
    k = len(decomp.sense_labels)
    
    categories = np.full(d, k + 1, dtype=int)
    for i, label in enumerate(decomp.sense_labels):
        for dim in decomp.unique_dimensions[label]:
            categories[dim] = i
    for dim in range(d):
        if decomp.contested_dimensions[dim]:
            categories[dim] = k + 2
    
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
        handles.append(mpatches.Patch(color=SENSE_COLORS[i % len(SENSE_COLORS)],
                                      label=f'{label} ({n})'))
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
    d = len(decomp.word_vector)
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
    k = len(decomp.sense_labels)
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
    
    # Word vector
    ax.annotate('', xy=proj[0], xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
    ax.text(proj[0, 0] * 1.08, proj[0, 1] * 1.08, 'w',
            fontsize=12, fontweight='bold', ha='center', va='center')
    
    # Sense vectors
    for i in range(k):
        c = SENSE_COLORS[i % len(SENSE_COLORS)]
        ax.annotate('', xy=proj[1+i], xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=2.0))
        ax.text(proj[1+i, 0] * 1.12, proj[1+i, 1] * 1.12,
                decomp.sense_labels[i], fontsize=10, color=c,
                fontweight='bold', ha='center', va='center')
    
    # Reconstruction
    ri = 1 + k
    ax.annotate('', xy=proj[ri], xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='grey', lw=1.5, linestyle='dashed'))
    ax.text(proj[ri, 0] * 1.12, proj[ri, 1] * 1.12, 'ŵ',
            fontsize=10, color='grey', ha='center', va='center')
    
    # Angle annotations
    for i in range(k):
        for j in range(i+1, k):
            angle = decomp.sense_angles_deg[i, j]
            mid = (proj[1+i] + proj[1+j]) / 2
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
    ax.set_title(title or f"Sense Geometry: '{decomp.word}'")
    ve = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ve[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({ve[1]:.0%} var)")


def plot_sign_interference(decomp, ax, title=None):
    k = len(decomp.sense_labels)
    d = len(decomp.word_vector)
    
    interference = np.zeros((k, d))
    for i in range(k):
        sig = np.abs(decomp.sense_contributions[i]) > 1e-6
        agrees = np.sign(decomp.sense_contributions[i]) == np.sign(decomp.word_vector)
        interference[i, sig & agrees] = 1.0
        interference[i, sig & ~agrees] = -1.0
    
    cmap = LinearSegmentedColormap.from_list('interf', ['#F44336', '#FFFFFF', '#2196F3'])
    im = ax.imshow(interference, aspect='auto', cmap=cmap, vmin=-1, vmax=1,
                   interpolation='nearest')
    ax.set_yticks(range(k))
    ax.set_yticklabels(decomp.sense_labels)
    ax.set_xlabel("Dimension index")
    ax.set_title(title or f"Interference: '{decomp.word}' (blue=construct, red=destruct)")
    plt.colorbar(im, ax=ax, shrink=0.8, ticks=[-1, 0, 1])


def plot_word_dashboard(decomp, save_path):
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3,
                          height_ratios=[1, 1, 1, 0.7, 0.7])
    
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
    plot_sign_interference(decomp, ax6)
    
    # Summary text
    ax7 = fig.add_subplot(gs[4, :])
    ax7.axis('off')
    k = len(decomp.sense_labels)
    lines = [
        f"Word: '{decomp.word}'   |   Senses: {k}   |   Dims: {len(decomp.word_vector)}",
        f"R² = {decomp.variance_explained_total:.4f}   |   "
        f"Residual ratio = {decomp.residual_norm_ratio:.4f}",
        "",
        "Coefficients: " + "  |  ".join(
            [f"α({decomp.sense_labels[i]}) = {decomp.coefficients[i]:.3f}" for i in range(k)]),
        "Word→sense cosines: " + "  |  ".join(
            [f"cos(w,{decomp.sense_labels[i]}) = {decomp.word_sense_cosines[i]:.3f}" for i in range(k)]),
    ]
    if k >= 2:
        angle_strs = []
        for i in range(k):
            for j in range(i+1, k):
                angle_strs.append(f"∠({decomp.sense_labels[i]},{decomp.sense_labels[j]}) = "
                                  f"{decomp.sense_angles_deg[i,j]:.1f}°")
        lines.append("Inter-sense angles: " + "  |  ".join(angle_strs))
    
    n_shared = int(np.sum(decomp.shared_dimensions))
    n_contested = int(np.sum(decomp.contested_dimensions))
    terr = [f"{l}: {len(decomp.unique_dimensions[l])} unique" for l in decomp.sense_labels]
    lines.append(f"Territories: " + "  |  ".join(terr) +
                 f"  |  shared: {n_shared}  |  contested: {n_contested}")
    
    ax7.text(0.02, 0.95, "\n".join(lines), transform=ax7.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='#CCCCCC'))
    
    fig.suptitle(f"Sense Localization Dashboard: '{decomp.word}'",
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_cross_word_comparison(decompositions, save_path):
    n = len(decompositions)
    fig, axes = plt.subplots(n, 3, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    
    for row, decomp in enumerate(decompositions):
        plot_dimension_attribution(decomp, axes[row, 0], sort_by_dominance=True,
                                   title=f"'{decomp.word}' — Attribution")
        plot_sense_territories(decomp, axes[row, 1],
                               title=f"'{decomp.word}' — Territories")
        plot_sense_geometry(decomp, axes[row, 2],
                            title=f"'{decomp.word}' — Geometry")
    
    fig.suptitle("Cross-Word Sense Localization Comparison",
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_angle_summary(decompositions, save_path):
    """Bar chart of all inter-sense angles — the key empirical result."""
    pairs = []
    for d in decompositions:
        k = len(d.sense_labels)
        for i in range(k):
            for j in range(i+1, k):
                label = f"{d.word}\n({d.sense_labels[i]}/{d.sense_labels[j]})"
                angle = d.sense_angles_deg[i, j]
                pairs.append((label, angle))
    
    labels, angles = zip(*pairs)
    
    fig, ax = plt.subplots(figsize=(max(12, len(pairs) * 1.2), 6))
    
    colors = ['#4CAF50' if 70 <= a <= 110 else '#F44336' for a in angles]
    bars = ax.bar(range(len(pairs)), angles, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=90, color='black', linewidth=2, linestyle='-', label='90° (orthogonal)')
    ax.axhspan(70, 110, alpha=0.1, color='green', label='Near-orthogonal zone (70-110°)')
    
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
    ax.set_ylabel("Inter-sense angle (degrees)")
    ax.set_title("ARE SENSES GEOMETRICALLY ORTHOGONAL IN GloVe?",
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 180)
    ax.legend(loc='upper right')
    
    # Add angle values on bars
    for i, (bar, angle) in enumerate(zip(bars, angles)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{angle:.0f}°', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    mean_angle = np.mean(angles)
    ax.axhline(y=mean_angle, color='blue', linewidth=1, linestyle='--',
               label=f'Mean = {mean_angle:.1f}°')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sense Vector Localization in GloVe')
    parser.add_argument('--glove', type=str, default='glove.6B.100d.txt',
                        help='Path to GloVe text file')
    parser.add_argument('--output', type=str, default='sense_localization_output',
                        help='Output directory for plots and reports')
    parser.add_argument('--words', type=str, nargs='*', default=None,
                        help='Specific words to analyze (default: all in SENSE_ANCHORS)')
    parser.add_argument('--use-sense-explorer', action='store_true',
                        help='Use SenseExplorer instead of standalone SSR')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON file with custom sense anchors. Format: '
                             '{"word": {"sense1": ["anchor1", ...], "sense2": [...]}}')
    args = parser.parse_args()
    
    # Load custom anchors from JSON if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            custom_anchors = json.load(f)
        # Merge: custom overrides built-in
        SENSE_ANCHORS.update(custom_anchors)
        for word in custom_anchors:
            EXPECTED_SENSES[word] = len(custom_anchors[word])
        print(f"Loaded custom anchors from {args.config}: "
              f"{list(custom_anchors.keys())}")
    
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 70)
    print("SENSE VECTOR LOCALIZATION — REAL GloVe EMBEDDINGS")
    print("=" * 70)
    
    # 1. Load embeddings
    embeddings = load_glove(args.glove)
    dim = len(next(iter(embeddings.values())))
    
    # 2. Select words to analyze
    words = args.words or list(SENSE_ANCHORS.keys())
    words = [w for w in words if w in embeddings]
    print(f"\nAnalyzing {len(words)} polysemous words: {words}")
    
    # 3. Extract sense vectors
    print(f"\n{'='*70}")
    print("PHASE 1: EXTRACTING SENSE VECTORS VIA SSR")
    print(f"{'='*70}")
    
    all_sense_vectors = {}  # {word: {sense: vector}}
    
    if args.use_sense_explorer:
        try:
            from sense_explorer import SenseExplorer
            print("Using SenseExplorer for sense extraction")
            se = SenseExplorer(embeddings)
            for word in words:
                print(f"\n  {word}:", end=" ")
                senses = se.induce_senses(word)
                if senses and len(senses) >= 2:
                    all_sense_vectors[word] = {k: v for k, v in senses.items()}
                    print(f"found {len(senses)} senses: {list(senses.keys())}")
                else:
                    print(f"only {len(senses) if senses else 0} sense(s), trying standalone SSR")
                    if word in SENSE_ANCHORS:
                        sv = run_ssr(word, embeddings, SENSE_ANCHORS[word])
                        if len(sv) >= 2:
                            all_sense_vectors[word] = sv
                            print(f"    SSR found {len(sv)} senses: {list(sv.keys())}")
        except ImportError:
            print("SenseExplorer not available, using standalone SSR")
            args.use_sense_explorer = False
    
    if not args.use_sense_explorer:
        print("Using standalone SSR")
        for word in words:
            if word not in SENSE_ANCHORS:
                print(f"\n  *** WARNING: '{word}' has no anchors defined in SENSE_ANCHORS. ***")
                print(f"      Add it to SENSE_ANCHORS in the script, or use --config with a JSON file.")
                print(f"      Skipping '{word}'.")
                continue
            print(f"\n  {word}:", end=" ")
            sv = run_ssr(word, embeddings, SENSE_ANCHORS[word])
            if len(sv) >= 2:
                all_sense_vectors[word] = sv
                print(f"found {len(sv)} senses: {list(sv.keys())}")
                # Print sense similarity
                labels = list(sv.keys())
                for i in range(len(labels)):
                    for j in range(i+1, len(labels)):
                        cos = np.dot(sv[labels[i]], sv[labels[j]]) / (
                            norm(sv[labels[i]]) * norm(sv[labels[j]]) + 1e-10)
                        print(f"    cos({labels[i]}, {labels[j]}) = {cos:.4f}")
            else:
                print(f"failed (only {len(sv)} sense)")
    
    if not all_sense_vectors:
        print("\nERROR: No sense vectors extracted. Check your GloVe file path.")
        sys.exit(1)
    
    # 4. Decompose and analyze
    print(f"\n{'='*70}")
    print("PHASE 2: SENSE LOCALIZATION")
    print(f"{'='*70}")
    
    decompositions = []
    for word, senses in all_sense_vectors.items():
        w = embeddings[word]
        decomp = decompose(word, w, senses)
        decompositions.append(decomp)
        print_report(decomp)
    
    # 5. Cross-word generalizations
    print_cross_word_summary(decompositions)
    
    # 6. Generate visualizations
    print(f"\n{'='*70}")
    print("PHASE 3: GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    for decomp in decompositions:
        path = os.path.join(args.output, f'dashboard_{decomp.word}.png')
        plot_word_dashboard(decomp, path)
    
    path = os.path.join(args.output, 'cross_word_comparison.png')
    plot_cross_word_comparison(decompositions, path)
    
    path = os.path.join(args.output, 'angle_summary.png')
    plot_angle_summary(decompositions, path)
    
    # 7. Save raw data for further analysis
    data_path = os.path.join(args.output, 'decomposition_data.npz')
    save_data = {}
    for decomp in decompositions:
        prefix = decomp.word
        save_data[f'{prefix}_word_vector'] = decomp.word_vector
        save_data[f'{prefix}_sense_vectors'] = decomp.sense_vectors
        save_data[f'{prefix}_coefficients'] = decomp.coefficients
        save_data[f'{prefix}_residual'] = decomp.residual
        save_data[f'{prefix}_angles'] = decomp.sense_angles_deg
        save_data[f'{prefix}_cosines'] = decomp.sense_cosine_matrix
        save_data[f'{prefix}_var_explained'] = np.array([decomp.variance_explained_total])
    np.savez(data_path, **save_data)
    print(f"\n  Raw data saved: {data_path}")
    
    # 8. Final summary
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    print(f"\nAll outputs in: {args.output}/")
    for f in sorted(os.listdir(args.output)):
        print(f"  {f}")
    
    all_angles = []
    for d in decompositions:
        k = len(d.sense_labels)
        for i in range(k):
            for j in range(i+1, k):
                all_angles.append(d.sense_angles_deg[i, j])
    
    if all_angles:
        print(f"\n*** KEY FINDING ***")
        print(f"Mean inter-sense angle: {np.mean(all_angles):.1f}° ± {np.std(all_angles):.1f}°")
        if np.mean(all_angles) > 70:
            print(f"→ Senses ARE geometrically separated in real GloVe embeddings!")
        else:
            print(f"→ Senses show less separation than expected. Investigate further.")


if __name__ == '__main__':
    main()
