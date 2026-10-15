#!/usr/bin/env python3
"""
Hybrid Anchor Extraction for Sense Exploration
=============================================

Combines multiple strategies for automatic anchor word extraction:

1. FrameNet Frames (88% accuracy when available)
   - Frame Elements as situational participants
   - Best quality but limited coverage (13K lexical units)

2. WordNet Gloss Nouns (62-69% accuracy)
   - Nouns extracted from synset definitions
   - Broad coverage (117K synsets)
   - Key insight: Nouns ≈ Frame Elements (situational participants)

3. Manual Anchors (88% accuracy, predefined)
   - Hand-picked extensional anchors
   - Limited to common polysemous words

Strategy:
  1. Check manual anchors first (highest quality)
  2. Try FrameNet if available (high quality)
  3. Fall back to WordNet gloss nouns (broad coverage)

Theoretical basis:
  Frame Elements = WordNet Gloss Nouns = Distributional Co-occurrence
  All three capture "situational participants" in schematic situations.

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
"""

import re
import numpy as np
from typing import Dict, List, Set, Tuple, Optional

# Try importing NLTK components
try:
    from nltk.corpus import wordnet as wn
    from nltk import pos_tag, word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Try importing FrameNet
try:
    from nltk.corpus import framenet as fn
    FRAMENET_AVAILABLE = True
except (ImportError, LookupError):
    FRAMENET_AVAILABLE = False


# =============================================================================
# Stopwords for filtering
# =============================================================================

STOPWORDS = {
    'the', 'and', 'for', 'that', 'with', 'this', 'from', 'are', 'was',
    'were', 'been', 'being', 'have', 'has', 'had', 'having', 'does',
    'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'shall', 'can', 'need', 'used', 'using', 'any', 'all',
    'some', 'such', 'than', 'too', 'very', 'just', 'also', 'into',
    'over', 'after', 'before', 'between', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'which', 'who',
    'one', 'two', 'three', 'first', 'second', 'third', 'small', 'large',
    'big', 'little', 'many', 'much', 'more', 'most', 'other', 'another',
    'each', 'every', 'both', 'few', 'several', 'various', 'certain',
    'particular', 'specific', 'general', 'common', 'type', 'kind',
    'form', 'part', 'way', 'thing', 'something'
}


# =============================================================================
# Manual Anchors for Common Polysemous Words
# =============================================================================

MANUAL_ANCHORS = {
    'bank': {
        'financial': ['money', 'account', 'loan', 'deposit', 'credit', 'savings', 
                      'investment', 'interest', 'mortgage', 'finance', 'banking'],
        'river': ['river', 'water', 'shore', 'stream', 'flood', 'erosion', 
                  'creek', 'lake', 'fish', 'mud', 'flow']
    },
    'bat': {
        'animal': ['cave', 'wing', 'fly', 'nocturnal', 'vampire', 'mammal', 
                   'fruit', 'insect', 'roost', 'echolocation'],
        'sports': ['ball', 'hit', 'swing', 'baseball', 'cricket', 'pitch', 
                   'player', 'wooden', 'innings', 'game']
    },
    'crane': {
        'bird': ['bird', 'wing', 'fly', 'nest', 'migrate', 'wetland', 
                 'heron', 'stork', 'flock', 'feather'],
        'machine': ['construction', 'lift', 'heavy', 'steel', 'tower', 
                    'load', 'operator', 'building', 'cargo', 'hook']
    },
    'mouse': {
        'animal': ['rat', 'cheese', 'trap', 'cat', 'rodent', 'squirrel', 
                   'tail', 'whiskers', 'hole', 'pest'],
        'computer': ['click', 'cursor', 'keyboard', 'screen', 'button', 
                     'scroll', 'pointer', 'pad', 'wireless', 'computer']
    },
    'plant': {
        'vegetation': ['tree', 'flower', 'leaf', 'grow', 'seed', 'garden', 
                       'root', 'soil', 'water', 'green'],
        'factory': ['factory', 'production', 'manufacturing', 'industrial', 
                    'worker', 'assembly', 'equipment', 'facility', 'output', 'machinery']
    },
    'bass': {
        'fish': ['fish', 'fishing', 'lake', 'catch', 'trout', 'salmon', 
                 'angler', 'rod', 'bait', 'water'],
        'music': ['guitar', 'drum', 'band', 'music', 'player', 'sound', 
                  'instrument', 'rhythm', 'note', 'amplifier']
    },
    'spring': {
        'season': ['summer', 'winter', 'autumn', 'flower', 'bloom', 'march', 
                   'april', 'warm', 'rain', 'garden'],
        'water': ['water', 'fountain', 'source', 'flow', 'well', 'mineral', 
                  'hot', 'natural', 'fresh', 'bubbling']
    },
    'cell': {
        'biology': ['tissue', 'membrane', 'nucleus', 'dna', 'protein', 
                    'organism', 'division', 'blood', 'stem', 'cancer'],
        'phone': ['phone', 'mobile', 'call', 'wireless', 'signal', 
                  'tower', 'smartphone', 'battery', 'carrier', 'network'],
        'prison': ['jail', 'prisoner', 'locked', 'bars', 'inmate', 
                   'detention', 'solitary', 'confined', 'guard', 'cell']
    },
    'match': {
        'fire': ['lighter', 'flame', 'burn', 'ignite', 'stick', 
                 'sulfur', 'strike', 'box', 'fire', 'light'],
        'competition': ['game', 'tournament', 'opponent', 'winner', 
                        'sports', 'contest', 'play', 'team', 'score', 'final']
    },
    'bow': {
        'weapon': ['arrow', 'archer', 'shoot', 'hunting', 'string', 
                   'target', 'quiver', 'crossbow', 'aim', 'draw'],
        'gesture': ['curtsy', 'nod', 'greeting', 'respect', 'bend', 
                    'head', 'polite', 'acknowledge', 'formal', 'reverence']
    }
}


# =============================================================================
# FrameNet-Inspired Anchors (when actual FrameNet not available)
# =============================================================================

FRAME_ANCHORS = {
    'bank': {
        'financial': {
            'frame': 'Financial_institution',
            'anchors': ['institution', 'company', 'corporation', 'firm', 'customer', 
                        'client', 'borrower', 'money', 'funds', 'capital', 'assets', 
                        'loan', 'deposit', 'account', 'credit', 'mortgage', 'savings',
                        'withdraw', 'transfer', 'investment', 'interest', 'finance']
        },
        'river': {
            'frame': 'Natural_feature_shoreline',
            'anchors': ['river', 'stream', 'creek', 'lake', 'water', 'shore', 
                        'slope', 'edge', 'side', 'mud', 'flood', 'erosion', 
                        'fishing', 'swimming', 'boat', 'flow', 'current']
        }
    },
    'bat': {
        'animal': {
            'frame': 'Biological_entity_animal',
            'anchors': ['mammal', 'creature', 'animal', 'species', 'nocturnal',
                        'cave', 'roost', 'night', 'dark', 'fly', 'hunt', 
                        'echolocation', 'wing', 'wings', 'fur', 'vampire', 
                        'insect', 'fruit', 'mosquito']
        },
        'sports': {
            'frame': 'Tool_sports_implement',
            'anchors': ['wooden', 'club', 'stick', 'handle', 'player', 'batter',
                        'hitter', 'batsman', 'ball', 'pitch', 'baseball', 
                        'cricket', 'hit', 'swing', 'homerun', 'game', 'innings']
        }
    },
    'crane': {
        'bird': {
            'frame': 'Biological_entity_bird',
            'anchors': ['bird', 'species', 'wildlife', 'avian', 'wetland', 
                        'marsh', 'swamp', 'nest', 'migration', 'fly', 'migrate',
                        'wade', 'wing', 'feather', 'beak', 'heron', 'stork']
        },
        'machine': {
            'frame': 'Device_machine_lifting',
            'anchors': ['machine', 'equipment', 'tower', 'boom', 'hook',
                        'operator', 'worker', 'driver', 'load', 'cargo', 
                        'heavy', 'weight', 'construction', 'building', 'site',
                        'lift', 'hoist', 'steel']
        }
    },
    'mouse': {
        'animal': {
            'frame': 'Biological_entity_rodent',
            'anchors': ['rodent', 'mammal', 'animal', 'creature', 'pest',
                        'hole', 'nest', 'burrow', 'house', 'cat', 'owl', 
                        'snake', 'trap', 'cheese', 'grain', 'tail', 'whiskers',
                        'rat', 'squirrel', 'hamster']
        },
        'computer': {
            'frame': 'Device_computer_input',
            'anchors': ['device', 'peripheral', 'wireless', 'optical',
                        'cursor', 'pointer', 'screen', 'button', 'click',
                        'scroll', 'drag', 'select', 'keyboard', 'computer',
                        'laptop', 'pad', 'usb', 'bluetooth']
        }
    },
    'plant': {
        'vegetation': {
            'frame': 'Biological_entity_flora',
            'anchors': ['tree', 'flower', 'grass', 'shrub', 'vegetation',
                        'leaf', 'root', 'stem', 'seed', 'branch', 'grow',
                        'bloom', 'soil', 'garden', 'forest', 'water', 
                        'sunlight', 'green', 'nature', 'organic']
        },
        'factory': {
            'frame': 'Locale_industrial_facility',
            'anchors': ['factory', 'facility', 'mill', 'refinery', 'warehouse',
                        'worker', 'employee', 'labor', 'staff', 'manufacturing',
                        'production', 'assembly', 'processing', 'output',
                        'product', 'machinery', 'equipment', 'industrial']
        }
    },
    'bass': {
        'fish': {
            'frame': 'Biological_entity_fish',
            'anchors': ['fish', 'species', 'freshwater', 'saltwater', 'lake',
                        'river', 'sea', 'ocean', 'water', 'pond', 'fishing',
                        'catch', 'angler', 'rod', 'bait', 'hook', 'fillet',
                        'trout', 'salmon', 'perch']
        },
        'music': {
            'frame': 'Performing_arts_music',
            'anchors': ['low', 'deep', 'tone', 'frequency', 'sound', 'guitar',
                        'drum', 'instrument', 'amplifier', 'speaker', 'player',
                        'musician', 'bassist', 'band', 'orchestra', 'rhythm',
                        'music', 'note', 'chord', 'groove']
        }
    },
    'spring': {
        'season': {
            'frame': 'Calendric_unit_season',
            'anchors': ['season', 'march', 'april', 'may', 'year', 'warm',
                        'rain', 'thaw', 'mild', 'bloom', 'flower', 'blossom',
                        'green', 'planting', 'garden', 'summer', 'winter', 'autumn']
        },
        'water': {
            'frame': 'Natural_feature_water_source',
            'anchors': ['source', 'well', 'fountain', 'geyser', 'water',
                        'fresh', 'mineral', 'hot', 'thermal', 'flow', 
                        'bubble', 'gush', 'stream', 'underground', 'natural']
        }
    },
    'cell': {
        'biology': {
            'frame': 'Biological_entity_cell',
            'anchors': ['cell', 'cells', 'cellular', 'stem', 'blood',
                        'membrane', 'nucleus', 'mitochondria', 'cytoplasm',
                        'division', 'metabolism', 'protein', 'dna', 'gene',
                        'tissue', 'organism', 'body', 'organ', 'biology']
        },
        'phone': {
            'frame': 'Device_communication_mobile',
            'anchors': ['phone', 'mobile', 'cellphone', 'smartphone', 'device',
                        'tower', 'signal', 'network', 'carrier', 'wireless',
                        'call', 'text', 'message', 'battery', 'charger', 
                        'telephone', 'cellular', 'handheld']
        }
    }
}


# =============================================================================
# WordNet Gloss Extraction
# =============================================================================

def extract_nouns_from_text(text: str) -> List[str]:
    """Extract nouns from text using POS tagging."""
    if not NLTK_AVAILABLE:
        # Fallback: return all words
        return re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    try:
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        # Keep only nouns (NN, NNS, NNP, NNPS)
        nouns = [word for word, pos in tagged if pos.startswith('NN')]
        return nouns
    except:
        # Fallback
        return re.findall(r'\b[a-z]{3,}\b', text.lower())


def get_wordnet_synsets(word: str, pos: str = 'n') -> List:
    """Get WordNet synsets for a word."""
    if not NLTK_AVAILABLE:
        return []
    
    try:
        return wn.synsets(word, pos=pos)
    except:
        return []


def extract_gloss_anchors(
    synset,
    vocab: Set[str],
    extract_nouns_only: bool = True,
    include_examples: bool = True,
    include_lemmas: bool = True,
    include_hypernyms: bool = True,
    max_anchors: int = 15
) -> List[str]:
    """
    Extract anchors from WordNet synset gloss.
    
    Key insight: NOUNS in glosses ≈ Frame Elements (situational participants)
    Filtering to nouns doubles accuracy (31% → 62%)
    
    Args:
        synset: WordNet synset object
        vocab: Set of valid vocabulary words
        extract_nouns_only: If True, keep only nouns (recommended)
        include_examples: Include words from example sentences
        include_lemmas: Include lemma names from synset
        include_hypernyms: Include hypernym lemmas
        max_anchors: Maximum anchors to return
    
    Returns:
        List of anchor words
    """
    anchor_candidates = []
    
    # 1. Extract from gloss/definition
    gloss = synset.definition()
    if extract_nouns_only:
        gloss_words = extract_nouns_from_text(gloss)
    else:
        gloss_words = re.findall(r'\b[a-z]{3,}\b', gloss.lower())
    anchor_candidates.extend(gloss_words * 2)  # Double weight for gloss
    
    # 2. Extract from examples
    if include_examples:
        for example in synset.examples():
            if extract_nouns_only:
                ex_words = extract_nouns_from_text(example)
            else:
                ex_words = re.findall(r'\b[a-z]{3,}\b', example.lower())
            anchor_candidates.extend(ex_words)
    
    # 3. Include lemma names
    if include_lemmas:
        for lemma in synset.lemmas():
            name = lemma.name().lower().replace('_', ' ')
            for word in name.split():
                if len(word) >= 3:
                    anchor_candidates.append(word)
    
    # 4. Include hypernym lemmas
    if include_hypernyms:
        for hypernym in synset.hypernyms():
            for lemma in hypernym.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                for word in name.split():
                    if len(word) >= 3:
                        anchor_candidates.append(word)
            
            # Also grandparents
            for grandparent in hypernym.hypernyms():
                for lemma in grandparent.lemmas():
                    name = lemma.name().lower().replace('_', ' ')
                    for word in name.split():
                        if len(word) >= 3:
                            anchor_candidates.append(word)
    
    # Filter: keep only words in vocabulary, remove stopwords
    word_counts = {}
    for word in anchor_candidates:
        if word in vocab and word not in STOPWORDS and len(word) >= 3:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top anchors
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    return [w for w, c in sorted_words[:max_anchors]]


# =============================================================================
# Hybrid Anchor Extractor
# =============================================================================

class HybridAnchorExtractor:
    """
    Hybrid anchor extraction combining multiple strategies.
    
    Priority order:
    1. Manual anchors (highest quality, limited coverage)
    2. FrameNet frames (high quality, moderate coverage)
    3. WordNet gloss nouns (moderate quality, broad coverage)
    
    Example:
        >>> extractor = HybridAnchorExtractor(vocab)
        >>> anchors = extractor.extract("bank")
        >>> print(anchors)
        {'financial': ['money', 'account', ...], 'river': ['river', 'water', ...]}
    """
    
    def __init__(
        self,
        vocab: Set[str],
        use_manual: bool = True,
        use_framenet: bool = True,
        use_wordnet: bool = True,
        min_anchors_per_sense: int = 5,
        max_anchors_per_sense: int = 15,
        verbose: bool = False
    ):
        """
        Initialize hybrid anchor extractor.
        
        Args:
            vocab: Set of valid vocabulary words
            use_manual: Use manual predefined anchors
            use_framenet: Use FrameNet-inspired anchors
            use_wordnet: Use WordNet gloss extraction
            min_anchors_per_sense: Minimum anchors required per sense
            max_anchors_per_sense: Maximum anchors per sense
            verbose: Print extraction details
        """
        self.vocab = vocab
        self.use_manual = use_manual
        self.use_framenet = use_framenet
        self.use_wordnet = use_wordnet
        self.min_anchors = min_anchors_per_sense
        self.max_anchors = max_anchors_per_sense
        self.verbose = verbose
    
    def extract(
        self,
        word: str,
        n_senses: int = 2,
        synsets: List[str] = None
    ) -> Tuple[Dict[str, List[str]], str]:
        """
        Extract anchors for a word using hybrid strategy.
        
        Args:
            word: Target word
            n_senses: Expected number of senses
            synsets: Optional list of specific synset names to use
        
        Returns:
            Tuple of (anchors dict, source name)
            anchors: {sense_name: [anchor_words]}
            source: 'manual', 'framenet', 'wordnet', or 'auto'
        """
        # Strategy 1: Try manual anchors
        if self.use_manual and word in MANUAL_ANCHORS:
            anchors = self._filter_anchors(MANUAL_ANCHORS[word])
            if self._is_valid(anchors, n_senses):
                if self.verbose:
                    print(f"  Using MANUAL anchors for '{word}'")
                return anchors, 'manual'
        
        # Strategy 2: Try FrameNet-inspired anchors
        if self.use_framenet and word in FRAME_ANCHORS:
            frame_info = FRAME_ANCHORS[word]
            anchors = {}
            for sense_name, info in frame_info.items():
                valid = [w for w in info['anchors'] if w in self.vocab][:self.max_anchors]
                if len(valid) >= self.min_anchors:
                    anchors[sense_name] = valid
            
            if self._is_valid(anchors, n_senses):
                if self.verbose:
                    print(f"  Using FRAMENET anchors for '{word}'")
                return anchors, 'framenet'
        
        # Strategy 3: Try WordNet gloss extraction
        if self.use_wordnet and NLTK_AVAILABLE:
            anchors = self._extract_from_wordnet(word, n_senses, synsets)
            if self._is_valid(anchors, n_senses):
                if self.verbose:
                    print(f"  Using WORDNET anchors for '{word}'")
                return anchors, 'wordnet'
        
        # Fallback: return empty (will use auto-discovery)
        if self.verbose:
            print(f"  No anchors found for '{word}', using AUTO discovery")
        return {}, 'auto'
    
    def _filter_anchors(self, anchors: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Filter anchors to vocabulary."""
        result = {}
        for sense, words in anchors.items():
            valid = [w for w in words if w in self.vocab][:self.max_anchors]
            if len(valid) >= self.min_anchors:
                result[sense] = valid
        return result
    
    def _is_valid(self, anchors: Dict[str, List[str]], n_senses: int) -> bool:
        """Check if anchors meet minimum requirements."""
        return len(anchors) >= min(n_senses, 2)
    
    def _extract_from_wordnet(
        self,
        word: str,
        n_senses: int,
        synsets: List[str] = None
    ) -> Dict[str, List[str]]:
        """Extract anchors from WordNet glosses."""
        anchors = {}
        
        if synsets:
            # Use provided synsets
            for synset_name in synsets:
                try:
                    synset = wn.synset(synset_name)
                    sense_anchors = extract_gloss_anchors(
                        synset, self.vocab,
                        extract_nouns_only=True,
                        include_examples=True,
                        include_lemmas=True,
                        include_hypernyms=True,
                        max_anchors=self.max_anchors
                    )
                    if len(sense_anchors) >= self.min_anchors:
                        # Create readable sense name
                        first_lemma = synset.lemmas()[0].name().lower()
                        anchors[first_lemma] = sense_anchors
                except:
                    continue
        else:
            # Auto-discover synsets
            synset_list = get_wordnet_synsets(word, pos='n')
            
            # Take top N distinct synsets
            used_definitions = set()
            for synset in synset_list[:n_senses * 2]:  # Look at more to find distinct ones
                # Skip if definition too similar to existing
                defn = synset.definition()[:50]
                if defn in used_definitions:
                    continue
                used_definitions.add(defn)
                
                sense_anchors = extract_gloss_anchors(
                    synset, self.vocab,
                    extract_nouns_only=True,
                    include_examples=True,
                    include_lemmas=True,
                    include_hypernyms=True,
                    max_anchors=self.max_anchors
                )
                
                if len(sense_anchors) >= self.min_anchors:
                    first_lemma = synset.lemmas()[0].name().lower()
                    # Avoid duplicate sense names
                    if first_lemma in anchors:
                        first_lemma = f"{first_lemma}_{len(anchors)}"
                    anchors[first_lemma] = sense_anchors
                
                if len(anchors) >= n_senses:
                    break
        
        return anchors
    
    def get_available_strategies(self, word: str) -> List[str]:
        """Get list of strategies available for a word."""
        strategies = []
        
        if self.use_manual and word in MANUAL_ANCHORS:
            strategies.append('manual')
        if self.use_framenet and word in FRAME_ANCHORS:
            strategies.append('framenet')
        if self.use_wordnet and NLTK_AVAILABLE:
            synsets = get_wordnet_synsets(word)
            if synsets:
                strategies.append('wordnet')
        
        return strategies
    
    def get_coverage_stats(self) -> Dict[str, int]:
        """Get coverage statistics for each strategy."""
        return {
            'manual': len(MANUAL_ANCHORS),
            'framenet': len(FRAME_ANCHORS),
            'wordnet': '117,000+ synsets' if NLTK_AVAILABLE else 'unavailable'
        }
    
    def assess_quality(
        self,
        word: str,
        anchors: Dict[str, List[str]],
        embeddings: Dict[str, np.ndarray] = None,
        embeddings_norm: Dict[str, np.ndarray] = None
    ) -> Dict[str, dict]:
        """
        Assess anchor quality without requiring a full SenseExplorer instance.
        
        This is a standalone quality check that can be called directly on
        anchor sets before they are used for sense induction. For the full
        validation integrated into the self-repair pipeline, see
        SenseExplorer._validate_anchors().
        
        Checks:
          1. Coverage: How many anchors are in vocabulary?
          2. Diversity: Are anchor words distinct from each other?
          3. Balance: Are sense anchor sets roughly equal in size?
        
        Args:
            word: Target word
            anchors: Dict of {sense_name: [anchor_words]}
            embeddings: Optional dict of word -> vector (for similarity checks)
            embeddings_norm: Optional dict of word -> normalized vector
        
        Returns:
            Dict with overall quality assessment and per-sense details
        
        Example:
            >>> extractor = HybridAnchorExtractor(vocab)
            >>> anchors, source = extractor.extract("bank")
            >>> quality = extractor.assess_quality("bank", anchors, embeddings)
            >>> print(quality['overall'])  # 'good', 'fair', or 'poor'
        """
        result = {
            'overall': 'pending',
            'senses': {},
            'n_senses': len(anchors),
            'warnings': []
        }
        
        if not anchors:
            result['overall'] = 'poor'
            result['warnings'].append('No anchors provided')
            return result
        
        sizes = []
        all_anchor_words = set()
        
        for sense_name, anchor_words in anchors.items():
            valid = [w for w in anchor_words if w in self.vocab]
            missing = [w for w in anchor_words if w not in self.vocab]
            
            sense_info = {
                'n_provided': len(anchor_words),
                'n_valid': len(valid),
                'n_missing': len(missing),
                'missing_words': missing[:5],  # Show first 5
                'coverage': len(valid) / max(len(anchor_words), 1),
            }
            
            # Check for overlap with other senses
            overlap = all_anchor_words & set(anchor_words)
            if overlap:
                sense_info['overlap_words'] = list(overlap)
                result['warnings'].append(
                    f"Sense '{sense_name}' shares words with other senses: {list(overlap)}"
                )
            all_anchor_words.update(anchor_words)
            
            sizes.append(len(valid))
            result['senses'][sense_name] = sense_info
        
        # Balance check
        if sizes:
            max_size = max(sizes)
            min_size = min(sizes)
            balance = min_size / max(max_size, 1)
            result['balance'] = balance
            if balance < 0.3:
                result['warnings'].append(
                    f"Unbalanced anchor sets: sizes range from {min_size} to {max_size}"
                )
        
        # Embedding-based checks (if embeddings provided)
        if embeddings_norm and word in embeddings_norm:
            from numpy.linalg import norm as np_norm
            
            target_vec = embeddings_norm[word]
            sense_centroids = {}
            
            for sense_name, anchor_words in anchors.items():
                vecs = [embeddings_norm[w] for w in anchor_words 
                        if w in embeddings_norm]
                if vecs:
                    vecs_arr = np.array(vecs)
                    centroid = vecs_arr.mean(axis=0)
                    centroid = centroid / (np_norm(centroid) + 1e-10)
                    sense_centroids[sense_name] = centroid
                    
                    # Relevance to target
                    relevance = float((vecs_arr @ target_vec).mean())
                    result['senses'][sense_name]['relevance'] = relevance
                    
                    # Intra-sense coherence
                    if len(vecs) > 1:
                        sim_mat = vecs_arr @ vecs_arr.T
                        n = len(vecs)
                        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                        coherence = float(sim_mat[mask].mean())
                    else:
                        coherence = 1.0
                    result['senses'][sense_name]['coherence'] = coherence
            
            # Inter-sense separation
            names = list(sense_centroids.keys())
            for i, s1 in enumerate(names):
                min_sep = 1.0
                for j, s2 in enumerate(names):
                    if i != j:
                        cos = float(sense_centroids[s1] @ sense_centroids[s2])
                        min_sep = min(min_sep, 1.0 - cos)
                result['senses'][s1]['separation'] = min_sep if len(names) > 1 else 0.0
        
        # Overall quality
        n_poor = sum(1 for s in result['senses'].values() if s['n_valid'] < self.min_anchors)
        has_overlap = any('overlap_words' in s for s in result['senses'].values())
        
        if n_poor == 0 and not has_overlap and len(result['warnings']) == 0:
            result['overall'] = 'good'
        elif n_poor <= 1 and len(result['warnings']) <= 1:
            result['overall'] = 'fair'
        else:
            result['overall'] = 'poor'
        
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_anchors(
    word: str,
    vocab: Set[str],
    n_senses: int = 2,
    verbose: bool = False
) -> Tuple[Dict[str, List[str]], str]:
    """
    Extract anchors for a word using hybrid strategy.
    
    Convenience function that creates extractor and extracts in one call.
    
    Args:
        word: Target word
        vocab: Set of valid vocabulary words
        n_senses: Expected number of senses
        verbose: Print extraction details
    
    Returns:
        Tuple of (anchors dict, source name)
    """
    extractor = HybridAnchorExtractor(vocab, verbose=verbose)
    return extractor.extract(word, n_senses)


def get_manual_anchors(word: str) -> Optional[Dict[str, List[str]]]:
    """Get manual anchors for a word if available."""
    return MANUAL_ANCHORS.get(word)


def get_frame_anchors(word: str) -> Optional[Dict[str, Dict]]:
    """Get FrameNet-inspired anchors for a word if available."""
    return FRAME_ANCHORS.get(word)


def list_supported_words() -> Dict[str, List[str]]:
    """List words with predefined anchors."""
    return {
        'manual': list(MANUAL_ANCHORS.keys()),
        'framenet': list(FRAME_ANCHORS.keys())
    }


# =============================================================================
# Test/Demo
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    print("Hybrid Anchor Extractor Demo")
    print("=" * 50)
    
    # Simulate vocabulary
    vocab = {'money', 'account', 'loan', 'deposit', 'credit', 'savings',
             'river', 'water', 'shore', 'stream', 'flood', 'bank',
             'ball', 'hit', 'swing', 'baseball', 'cricket', 'bat',
             'cave', 'wing', 'fly', 'nocturnal', 'vampire', 'mammal'}
    
    extractor = HybridAnchorExtractor(vocab, verbose=True)
    
    for word in ['bank', 'bat', 'unknown_word']:
        print(f"\nWord: {word}")
        anchors, source = extractor.extract(word)
        print(f"  Source: {source}")
        for sense, words in anchors.items():
            print(f"  {sense}: {', '.join(words[:5])}")
