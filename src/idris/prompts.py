"""
Prompt generation for LLVM IR fuzzing.

This module handles seed collection and prompt creation using the new
parser and mutator infrastructure.
"""

import random
from pathlib import Path
from typing import Optional

from idris.parser import (
    ParsedFunction,
    parse_functions_from_directory,
    parse_functions_from_file,
)
from idris.mutator import (
    CommentMutator,
    FunctionClusterer,
    MutationStrategy,
    MutationResult,
    create_mutator,
)


def collect_seeds(test_dir: Path, 
                  max_count: int = 10000,
                  min_length: int = 100,
                  max_length: int = 2000,
                  vector_only: bool = False) -> list[ParsedFunction]:
    """
    Collect seed functions from LLVM test directory.
    
    Args:
        test_dir: Path to LLVM test directory
        max_count: Maximum number of functions to collect
        min_length: Minimum function length in characters
        max_length: Maximum function length in characters
        vector_only: If True, only collect functions with vector operations
    
    Returns:
        List of ParsedFunction objects
    """
    all_functions = parse_functions_from_directory(test_dir, max_count=max_count * 2)
    
    # Filter by length
    filtered = [
        f for f in all_functions 
        if min_length < len(f.raw_text) < max_length
    ]
    
    # Optionally filter to vector-only
    if vector_only:
        filtered = [f for f in filtered if f.has_vectors or f.has_scalable_vectors]
    
    return filtered[:max_count]


def truncate_randomly(text: str, min_ratio: float = 0.3, max_ratio: float = 0.8) -> str:
    """
    Truncate text at a random point for completion.
    
    Tries to cut at a newline for cleaner prompts.
    """
    ratio = random.uniform(min_ratio, max_ratio)
    cut_point = int(len(text) * ratio)
    
    # Try to find a newline near the cut point
    newline_pos = text.rfind('\n', 0, cut_point)
    if newline_pos > 50:
        return text[:newline_pos + 1]
    
    return text[:cut_point]


class PromptGenerator:
    """
    Generates prompts for the LLM using various mutation strategies.
    """
    
    def __init__(self, 
                 seeds: list[ParsedFunction],
                 seed_scores: Optional[list[tuple[ParsedFunction, float]]] = None,
                 weird_ratio: float = 0.3,
                 prefer_weird_prob: float = 0.5,
                 mutation_prob: float = 0.4,
                 truncate_min: float = 0.3,
                 truncate_max: float = 0.8):
        """
        Initialize the prompt generator.
        
        Args:
            seeds: List of seed functions
            seed_scores: Optional list of (seed, perplexity) tuples from scoring
            weird_ratio: Top fraction of seeds (by perplexity) to consider "weird"
            prefer_weird_prob: Probability of selecting from weird seeds vs normal
            mutation_prob: Probability of applying a mutation (vs just renaming)
            truncate_min: Minimum truncation ratio
            truncate_max: Maximum truncation ratio
        """
        self.seeds = seeds
        self.mutation_prob = mutation_prob
        self.truncate_min = truncate_min
        self.truncate_max = truncate_max
        self.prefer_weird_prob = prefer_weird_prob
        
        # Build clusterer and mutator
        self.clusterer = FunctionClusterer(seeds)
        self.mutator = CommentMutator(self.clusterer)
        
        # Separate pools for targeted selection
        self.vector_seeds = [s for s in seeds if s.has_vectors or s.has_scalable_vectors]
        self.memory_seeds = [s for s in seeds if s.has_memory_ops]
        self.control_flow_seeds = [s for s in seeds if s.has_control_flow]
        
        # Perplexity-based pools
        if seed_scores is not None and len(seed_scores) > 0:
            n_weird = max(1, int(len(seed_scores) * weird_ratio))
            self.weird_seeds = [s for s, _ in seed_scores[:n_weird]]
            self.normal_seeds = [s for s, _ in seed_scores[n_weird:]]
            self.has_scoring = True
            
            # Store perplexity lookup for metadata
            self.seed_perplexity = {id(s): ppl for s, ppl in seed_scores}
        else:
            self.weird_seeds = []
            self.normal_seeds = []
            self.has_scoring = False
            self.seed_perplexity = {}
    
    def generate_prompt(self, 
                        prefer_vectors: bool = False,
                        prefer_weird: Optional[bool] = None,
                        strategy: Optional[MutationStrategy] = None) -> tuple[str, dict]:
        """
        Generate a single prompt.
        
        Args:
            prefer_vectors: If True, prefer selecting vector functions
            prefer_weird: If True, select from weird seeds. If None, random based on prefer_weird_prob
            strategy: Force a specific mutation strategy (None for random)
        
        Returns:
            Tuple of (prompt_text, metadata_dict)
        """
        # Decide whether to use weird seed
        if prefer_weird is None and self.has_scoring:
            prefer_weird = random.random() < self.prefer_weird_prob
        
        # Select seed
        is_weird_seed = False
        if prefer_weird and self.weird_seeds:
            seed = random.choice(self.weird_seeds)
            is_weird_seed = True
        elif prefer_vectors and self.vector_seeds:
            seed = random.choice(self.vector_seeds)
        elif self.has_scoring and self.normal_seeds:
            seed = random.choice(self.normal_seeds)
        else:
            seed = random.choice(self.seeds)
        
        # Apply mutation
        result = self.mutator.mutate(
            seed, 
            strategy=strategy,
            mutation_prob=self.mutation_prob
        )
        
        # Truncate for completion
        prompt = truncate_randomly(
            result.text, 
            self.truncate_min, 
            self.truncate_max
        )
        
        # Get seed perplexity if available
        seed_perplexity = self.seed_perplexity.get(id(seed), None)
        
        metadata = {
            "original_name": result.original_name,
            "new_name": result.new_name,
            "strategy": result.strategy.value,
            "donor_name": result.donor_name,
            "seed_has_vectors": seed.has_vectors,
            "seed_has_memory": seed.has_memory_ops,
            "seed_has_control_flow": seed.has_control_flow,
            "full_length": len(result.text),
            "prompt_length": len(prompt),
            # New fields for perplexity experiment
            "is_weird_seed": is_weird_seed,
            "seed_perplexity": seed_perplexity,
        }
        
        return prompt, metadata
    
    def generate_batch(self, 
                       batch_size: int,
                       vector_ratio: float = 0.3) -> list[tuple[str, dict]]:
        """
        Generate a batch of prompts.
        
        Args:
            batch_size: Number of prompts to generate
            vector_ratio: Fraction of prompts that should prefer vector functions
        
        Returns:
            List of (prompt_text, metadata_dict) tuples
        """
        results = []
        n_vector = int(batch_size * vector_ratio)
        
        for i in range(batch_size):
            prefer_vectors = i < n_vector
            prompt, meta = self.generate_prompt(prefer_vectors=prefer_vectors)
            results.append((prompt, meta))
        
        return results
    
    def stats(self) -> dict:
        """Return statistics about the seed pool and clusters."""
        cluster_stats = self.clusterer.stats()
        return {
            **cluster_stats,
            "vector_seeds": len(self.vector_seeds),
            "memory_seeds": len(self.memory_seeds),
            "control_flow_seeds": len(self.control_flow_seeds),
            "has_scoring": self.has_scoring,
            "weird_seeds": len(self.weird_seeds),
            "normal_seeds": len(self.normal_seeds),
        }


# Legacy function for backwards compatibility with existing code
def extract_functions(test_dir: Path, max_count: int = 8000) -> list[str]:
    """
    Extract functions as raw strings (legacy interface).
    
    For new code, use collect_seeds() and PromptGenerator instead.
    """
    functions = parse_functions_from_directory(test_dir, max_count=max_count)
    
    # Generate new names and return raw text
    results = []
    for func in functions:
        import uuid
        unique_id = uuid.uuid4().hex[:6]
        new_name = f"@fuzz_target_{unique_id}"
        # Replace the function name everywhere (including comments)
        renamed = func.raw_text.replace(func.name, new_name)
        results.append(renamed)
    
    return results