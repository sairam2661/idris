"""
Comment Mutator for LLVM IR Fuzzing

Provides strategies for mutating FileCheck directives to create diverse
prompts that may lead to interesting LLM completions.
"""

import random
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from enum import Enum

from idris.parser import ParsedFunction, CheckBlock, CheckDirective


class MutationStrategy(Enum):
    NONE = "none"
    SWAP_CHECKS = "swap_checks"           # Replace all checks with another function's
    PARTIAL_SWAP = "partial_swap"         # Swap some check blocks, keep others
    PERTURB_VALUES = "perturb_values"     # Modify constants/names in checks
    INJECT_VECTOR = "inject_vector"       # Add vector checks to scalar functions
    STRIP_CHECKS = "strip_checks"         # Remove all checks (baseline)
    MIX_PREFIXES = "mix_prefixes"         # Combine checks from different test configs


@dataclass
class MutationResult:
    """Result of applying a mutation"""
    text: str
    strategy: MutationStrategy
    original_name: str
    new_name: str
    donor_name: Optional[str] = None      # If we borrowed from another function


def compute_cluster_key(func: ParsedFunction) -> str:
    """
    Create a clustering key based on function 'shape'.
    Functions with the same key are similar enough for comment swapping.
    """
    features = []
    
    # Broad return type category
    ret = func.return_type.lower()
    if 'void' in ret:
        features.append("ret:void")
    elif '<' in ret and 'x' in ret:
        features.append("ret:vec")
    elif 'ptr' in ret or '*' in ret:
        features.append("ret:ptr")
    elif any(t in ret for t in ['i1', 'i8', 'i16', 'i32', 'i64', 'i128']):
        features.append("ret:int")
    elif any(t in ret for t in ['float', 'double', 'half', 'fp128']):
        features.append("ret:fp")
    else:
        features.append("ret:other")
    
    # Argument count bucket
    n_args = len(func.arg_types)
    if n_args == 0:
        features.append("args:0")
    elif n_args <= 2:
        features.append("args:1-2")
    elif n_args <= 4:
        features.append("args:3-4")
    else:
        features.append("args:5+")
    
    # Operation categories
    if func.has_vectors or func.has_scalable_vectors:
        features.append("vec")
    if func.has_memory_ops:
        features.append("mem")
    if func.has_control_flow:
        features.append("cf")
    if func.has_calls:
        features.append("call")
    if func.has_atomics:
        features.append("atomic")
    
    return "|".join(sorted(features))


class FunctionClusterer:
    """Groups functions by similarity for targeted mutations"""
    
    def __init__(self, functions: list[ParsedFunction]):
        self.functions = functions
        self.clusters: dict[str, list[ParsedFunction]] = defaultdict(list)
        self.vector_functions: list[ParsedFunction] = []
        self.functions_with_checks: list[ParsedFunction] = []
        
        self._build_clusters()
    
    def _build_clusters(self):
        """Build all cluster indices"""
        for func in self.functions:
            key = compute_cluster_key(func)
            self.clusters[key].append(func)
            
            if func.has_vectors or func.has_scalable_vectors:
                self.vector_functions.append(func)
            
            if func.check_blocks:
                self.functions_with_checks.append(func)
    
    def get_similar(self, func: ParsedFunction, exclude_self: bool = True) -> list[ParsedFunction]:
        """Get functions in the same cluster"""
        key = compute_cluster_key(func)
        candidates = self.clusters.get(key, [])
        
        if exclude_self:
            candidates = [f for f in candidates if f.name != func.name]
        
        return candidates
    
    def get_vector_donor(self, exclude: Optional[str] = None) -> Optional[ParsedFunction]:
        """Get a random vector function for check injection"""
        candidates = self.vector_functions
        if exclude:
            candidates = [f for f in candidates if f.name != exclude]
        
        # Prefer functions with CHECK blocks
        with_checks = [f for f in candidates if f.check_blocks]
        if with_checks:
            return random.choice(with_checks)
        
        return random.choice(candidates) if candidates else None
    
    def get_random_with_checks(self, exclude: Optional[str] = None) -> Optional[ParsedFunction]:
        """Get any random function that has CHECK directives"""
        candidates = self.functions_with_checks
        if exclude:
            candidates = [f for f in candidates if f.name != exclude]
        
        return random.choice(candidates) if candidates else None
    
    def stats(self) -> dict:
        """Return clustering statistics"""
        return {
            "total_functions": len(self.functions),
            "num_clusters": len(self.clusters),
            "vector_functions": len(self.vector_functions),
            "functions_with_checks": len(self.functions_with_checks),
            "cluster_sizes": {k: len(v) for k, v in self.clusters.items()},
        }


class CommentMutator:
    """Applies various mutation strategies to LLVM IR functions"""
    
    def __init__(self, clusterer: FunctionClusterer):
        self.clusterer = clusterer
    
    def generate_new_name(self) -> str:
        """Generate a unique function name"""
        unique_id = uuid.uuid4().hex[:6]
        return f"@fuzz_target_{unique_id}"
    
    def mutate(self, func: ParsedFunction, 
               strategy: Optional[MutationStrategy] = None,
               mutation_prob: float = 0.5) -> MutationResult:
        """
        Apply a mutation strategy to a function.
        
        If strategy is None, randomly selects one (or none with 1-mutation_prob).
        """
        new_name = self.generate_new_name()
        
        # Maybe skip mutation
        if strategy is None:
            if random.random() > mutation_prob:
                strategy = MutationStrategy.NONE
            else:
                # Weight strategies by usefulness
                weights = [
                    (MutationStrategy.SWAP_CHECKS, 3),
                    (MutationStrategy.PARTIAL_SWAP, 2),
                    (MutationStrategy.PERTURB_VALUES, 2),
                    (MutationStrategy.INJECT_VECTOR, 2),
                    (MutationStrategy.STRIP_CHECKS, 1),
                    (MutationStrategy.MIX_PREFIXES, 1),
                ]
                strategies, probs = zip(*weights)
                strategy = random.choices(strategies, weights=probs)[0]
        
        # Apply the selected strategy
        if strategy == MutationStrategy.NONE:
            return self._no_mutation(func, new_name)
        elif strategy == MutationStrategy.SWAP_CHECKS:
            return self._swap_checks(func, new_name)
        elif strategy == MutationStrategy.PARTIAL_SWAP:
            return self._partial_swap(func, new_name)
        elif strategy == MutationStrategy.PERTURB_VALUES:
            return self._perturb_values(func, new_name)
        elif strategy == MutationStrategy.INJECT_VECTOR:
            return self._inject_vector_checks(func, new_name)
        elif strategy == MutationStrategy.STRIP_CHECKS:
            return self._strip_checks(func, new_name)
        elif strategy == MutationStrategy.MIX_PREFIXES:
            return self._mix_prefixes(func, new_name)
        else:
            return self._no_mutation(func, new_name)
    
    def _no_mutation(self, func: ParsedFunction, new_name: str) -> MutationResult:
        """Just rename, no other changes"""
        text = func.raw_text.replace(func.name, new_name)
        return MutationResult(
            text=text,
            strategy=MutationStrategy.NONE,
            original_name=func.name,
            new_name=new_name,
        )
    
    def _swap_checks(self, func: ParsedFunction, new_name: str) -> MutationResult:
        """Replace all CHECK blocks with those from a similar function"""
        similar = self.clusterer.get_similar(func)
        
        # Filter to those with checks
        with_checks = [f for f in similar if f.check_blocks]
        
        if not with_checks:
            # Fallback: try any function with checks
            donor = self.clusterer.get_random_with_checks(exclude=func.name)
            if not donor:
                return self._no_mutation(func, new_name)
        else:
            donor = random.choice(with_checks)
        
        # Rebuild with donor's checks
        text = func.with_checks(donor.check_blocks, new_name)
        
        return MutationResult(
            text=text,
            strategy=MutationStrategy.SWAP_CHECKS,
            original_name=func.name,
            new_name=new_name,
            donor_name=donor.name,
        )
    
    def _partial_swap(self, func: ParsedFunction, new_name: str) -> MutationResult:
        """Keep some original checks, swap others"""
        if not func.check_blocks:
            return self._swap_checks(func, new_name)
        
        donor = self.clusterer.get_random_with_checks(exclude=func.name)
        if not donor or not donor.check_blocks:
            return self._no_mutation(func, new_name)
        
        # Mix check blocks: take first half from original, second half from donor
        n_orig = len(func.check_blocks)
        n_donor = len(donor.check_blocks)
        
        split_orig = n_orig // 2
        split_donor = n_donor // 2
        
        mixed_checks = func.check_blocks[:split_orig] + donor.check_blocks[split_donor:]
        
        text = func.with_checks(mixed_checks, new_name)
        
        return MutationResult(
            text=text,
            strategy=MutationStrategy.PARTIAL_SWAP,
            original_name=func.name,
            new_name=new_name,
            donor_name=donor.name,
        )
    
    def _perturb_values(self, func: ParsedFunction, new_name: str) -> MutationResult:
        """Modify constants and capture names in CHECK directives"""
        if not func.check_blocks:
            return self._no_mutation(func, new_name)
        
        perturbed_blocks = []
        
        for block in func.check_blocks:
            new_directives = []
            for directive in block.directives:
                new_line = self._perturb_check_line(directive.raw_line)
                new_directives.append(CheckDirective(
                    prefix=directive.prefix,
                    kind=directive.kind,
                    content=directive.content,  # Keep original for reference
                    raw_line=new_line,
                ))
            perturbed_blocks.append(CheckBlock(directives=new_directives))
        
        text = func.with_checks(perturbed_blocks, new_name)
        
        return MutationResult(
            text=text,
            strategy=MutationStrategy.PERTURB_VALUES,
            original_name=func.name,
            new_name=new_name,
        )
    
    def _perturb_check_line(self, line: str) -> str:
        """Apply perturbations to a single CHECK line"""
        result = line
        
        # Perturb integer constants (e.g., i32 42 -> i32 43)
        def perturb_int(match):
            val = int(match.group(1))
            # Small perturbation
            delta = random.choice([-1, 1, -2, 2])
            return f" {val + delta}"
        
        result = re.sub(r'\s(\d+)(?=\s|,|\)|\]|$)', perturb_int, result)
        
        # Perturb capture names (e.g., [[VAR:%.*]] -> [[VAR2:%.*]])
        def perturb_capture(match):
            name = match.group(1)
            suffix = random.choice(['2', '_new', '_mut', ''])
            return f"[[{name}{suffix}:%.*]]"
        
        result = re.sub(r'\[\[(\w+):%\.\*\]\]', perturb_capture, result)
        
        return result
    
    def _inject_vector_checks(self, func: ParsedFunction, new_name: str) -> MutationResult:
        """Add vector-related CHECK comments to a (possibly scalar) function"""
        donor = self.clusterer.get_vector_donor(exclude=func.name)
        
        if not donor or not donor.check_blocks:
            return self._no_mutation(func, new_name)
        
        # Combine: original checks (if any) + vector checks from donor
        combined_checks = list(func.check_blocks) + list(donor.check_blocks)
        
        text = func.with_checks(combined_checks, new_name)
        
        return MutationResult(
            text=text,
            strategy=MutationStrategy.INJECT_VECTOR,
            original_name=func.name,
            new_name=new_name,
            donor_name=donor.name,
        )
    
    def _strip_checks(self, func: ParsedFunction, new_name: str) -> MutationResult:
        """Remove all CHECK directives, leave pure code"""
        text = func.with_checks([], new_name)
        
        return MutationResult(
            text=text,
            strategy=MutationStrategy.STRIP_CHECKS,
            original_name=func.name,
            new_name=new_name,
        )
    
    def _mix_prefixes(self, func: ParsedFunction, new_name: str) -> MutationResult:
        """Combine CHECK blocks from functions with different test prefixes"""
        if not func.check_blocks:
            return self._swap_checks(func, new_name)
        
        # Find our prefixes
        our_prefixes = set()
        for block in func.check_blocks:
            our_prefixes.update(block.prefixes)
        
        # Find a function with different prefixes
        candidates = self.clusterer.functions_with_checks
        different_prefix_funcs = []
        
        for candidate in candidates:
            if candidate.name == func.name:
                continue
            cand_prefixes = set()
            for block in candidate.check_blocks:
                cand_prefixes.update(block.prefixes)
            
            if cand_prefixes and cand_prefixes != our_prefixes:
                different_prefix_funcs.append(candidate)
        
        if not different_prefix_funcs:
            return self._no_mutation(func, new_name)
        
        donor = random.choice(different_prefix_funcs)
        
        # Combine both sets of checks
        combined = list(func.check_blocks) + list(donor.check_blocks)
        
        text = func.with_checks(combined, new_name)
        
        return MutationResult(
            text=text,
            strategy=MutationStrategy.MIX_PREFIXES,
            original_name=func.name,
            new_name=new_name,
            donor_name=donor.name,
        )


def create_mutator(functions: list[ParsedFunction]) -> CommentMutator:
    """Convenience function to create a mutator from a list of functions"""
    clusterer = FunctionClusterer(functions)
    return CommentMutator(clusterer)