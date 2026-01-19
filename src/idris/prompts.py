"""
Pattern-guided prompt generation for LLVM IR fuzzing.

This module implements a cleaner approach: instead of mutating FileCheck
directives (which adds noise), we extract semantic patterns from test cases
and generate prompts that guide the LLM to produce interesting IR.

Key insight: bugs tend to come from minimal prompts (just signatures).
FileCheck directives confuse the model into trying to match expected output
rather than generating novel, potentially buggy IR.
"""

import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum

from idris.parser import (
    ParsedFunction,
    parse_functions_from_directory,
    VECTOR_TYPE_PATTERN,
    SCALABLE_VECTOR_PATTERN,
)


class PromptStrategy(Enum):
    """Different strategies for generating prompts."""
    # Baseline strategies
    SIGNATURE_ONLY = "signature_only"           # Just the define line
    SIGNATURE_WITH_ENTRY = "signature_entry"    # Define + entry block start
    
    # Pattern-guided strategies  
    PATTERN_GUIDED = "pattern_guided"           # Signature + semantic hint
    TYPE_NOVEL = "type_novel"                   # Known pattern + unusual types
    
    # For comparison with old approach
    TRUNCATED_FULL = "truncated_full"           # Old style: truncate full function


@dataclass
class TransformationPattern:
    """A pattern describing what kind of transformation a test exercises."""
    name: str
    description: str
    input_hints: list[str]      # What the input IR typically contains
    output_hints: list[str]     # What the optimized IR typically contains
    relevant_passes: list[str]  # Which passes this pattern relates to
    example_signature: Optional[str] = None


# Common transformation patterns extracted from LLVM test suite structure
TRANSFORMATION_PATTERNS = [
    TransformationPattern(
        name="constant_folding",
        description="Operations on constants that can be computed at compile time",
        input_hints=["arithmetic on literals", "comparison with constants"],
        output_hints=["single constant return", "simplified expression"],
        relevant_passes=["instcombine", "constprop"],
    ),
    TransformationPattern(
        name="dead_code",
        description="Code that doesn't affect the output",
        input_hints=["unused computations", "unreachable blocks"],
        output_hints=["removed instructions", "simplified CFG"],
        relevant_passes=["dce", "adce", "simplifycfg"],
    ),
    TransformationPattern(
        name="strength_reduction",
        description="Replace expensive ops with cheaper equivalents",
        input_hints=["multiply by power of 2", "divide by constant"],
        output_hints=["shift operations", "bitwise ops"],
        relevant_passes=["instcombine"],
    ),
    TransformationPattern(
        name="redundant_load_store",
        description="Memory operations that can be eliminated",
        input_hints=["store followed by load", "repeated loads"],
        output_hints=["forwarded values", "eliminated memory ops"],
        relevant_passes=["gvn", "dse", "memcpyopt"],
    ),
    TransformationPattern(
        name="loop_invariant",
        description="Computations that don't change across loop iterations",
        input_hints=["computation inside loop", "no loop-carried dependency"],
        output_hints=["hoisted computation", "simplified loop body"],
        relevant_passes=["licm", "loop-unroll"],
    ),
    TransformationPattern(
        name="algebraic_simplify",
        description="Mathematical identities and simplifications",
        input_hints=["x + 0", "x * 1", "x - x", "x & -1"],
        output_hints=["identity removed", "simplified to operand"],
        relevant_passes=["instcombine", "reassociate"],
    ),
    TransformationPattern(
        name="branch_folding",
        description="Conditional branches that can be resolved",
        input_hints=["branch on constant", "redundant conditions"],
        output_hints=["unconditional branch", "removed block"],
        relevant_passes=["simplifycfg", "jump-threading"],
    ),
    TransformationPattern(
        name="vector_combine",
        description="Scalar ops that can be vectorized or vector ops that can be simplified",
        input_hints=["repeated scalar ops", "extractelement/insertelement chains"],
        output_hints=["vector operation", "simplified shuffle"],
        relevant_passes=["slp-vectorizer", "instcombine"],
    ),
    TransformationPattern(
        name="phi_simplify",
        description="PHI nodes that can be simplified",
        input_hints=["phi with identical operands", "single predecessor phi"],
        output_hints=["removed phi", "direct value use"],
        relevant_passes=["instcombine", "simplifycfg"],
    ),
    TransformationPattern(
        name="call_simplify",
        description="Function calls that can be simplified or folded",
        input_hints=["intrinsic with constant args", "known library call"],
        output_hints=["constant result", "simpler intrinsic"],
        relevant_passes=["instcombine"],
    ),
]

# Interesting types that may expose edge cases
INTERESTING_TYPES = {
    "scalable_vectors": [
        "<vscale x 1 x i1>",
        "<vscale x 2 x i1>",
        "<vscale x 4 x i1>",
        "<vscale x 2 x i8>",
        "<vscale x 4 x i8>",
        "<vscale x 2 x i16>",
        "<vscale x 4 x i16>",
        "<vscale x 2 x i32>",
        "<vscale x 4 x i32>",
        "<vscale x 2 x i64>",
        "<vscale x 2 x float>",
        "<vscale x 4 x float>",
        "<vscale x 2 x double>",
        "<vscale x 2 x ptr>",
    ],
    "fixed_vectors": [
        "<2 x i1>",
        "<4 x i1>",
        "<8 x i1>",
        "<3 x i32>",   # Non-power-of-2
        "<5 x i32>",
        "<7 x float>",
        "<16 x i8>",
        "<32 x i8>",
        "<2 x i128>",
        "<4 x half>",
        "<2 x bfloat>",
    ],
    "unusual_integers": [
        "i1", "i3", "i5", "i7",  # Small odd widths
        "i128", "i256",          # Large widths
        "i17", "i33", "i65",     # Uncommon widths
    ],
    "floats": [
        "half",
        "bfloat",
        "float",
        "double",
        "fp128",
    ],
    "pointers": [
        "ptr",
        "ptr addrspace(1)",
        "ptr addrspace(3)",
    ],
}


@dataclass
class PromptResult:
    """Result of generating a prompt."""
    text: str
    strategy: PromptStrategy
    metadata: dict = field(default_factory=dict)


def generate_function_name() -> str:
    """Generate a unique function name."""
    unique_id = uuid.uuid4().hex[:8]
    return f"@fuzz_{unique_id}"


def strip_filecheck_comments(text: str) -> str:
    """Remove all FileCheck directives from text."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # Skip CHECK lines
        if re.match(r'\s*;\s*\w+-?(LABEL|NEXT|SAME|NOT|DAG|CHECK|COUNT)', line):
            continue
        # Skip RUN lines
        if re.match(r'\s*;\s*RUN:', line):
            continue
        # Skip NOTE lines about autogenerated checks
        if 'Assertions have been autogenerated' in line:
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


def extract_signature_only(func: ParsedFunction) -> str:
    """Extract just the function signature (define line)."""
    return func.signature


def extract_clean_body(func: ParsedFunction) -> list[str]:
    """Extract body lines without CHECK comments."""
    cleaned = []
    for line in func.body_lines:
        if re.match(r'\s*;\s*\w+-?(LABEL|NEXT|SAME|NOT|DAG|CHECK|COUNT)', line):
            continue
        cleaned.append(line)
    return cleaned


def substitute_types_in_signature(signature: str, new_types: list[str]) -> str:
    """
    Replace types in a signature with new interesting types.
    Tries to maintain structural validity.
    """
    result = signature
    
    # Find existing vector types and replace with scalable vectors
    if random.random() < 0.5:
        # Replace fixed vectors with scalable
        def replace_fixed_vector(match):
            count = match.group(1)
            elem_type = match.group(2)
            # Try to make it scalable
            new_count = random.choice(['1', '2', '4'])
            return f"<vscale x {new_count} x {elem_type}>"
        
        result = re.sub(r'<(\d+)\s*x\s*(\w+)>', replace_fixed_vector, result)
    
    # Maybe replace integer types with unusual widths
    if random.random() < 0.3:
        unusual_int = random.choice(INTERESTING_TYPES["unusual_integers"])
        result = re.sub(r'\bi32\b', unusual_int, result, count=1)
    
    return result


class PatternGuidedGenerator:
    """
    Generates prompts using semantic patterns rather than FileCheck mutation.
    """
    
    def __init__(
        self,
        seeds: list[ParsedFunction],
        seed_scores: Optional[list[tuple[ParsedFunction, float]]] = None,
        weird_ratio: float = 0.3,
        prefer_weird_prob: float = 0.5,
    ):
        self.seeds = seeds
        self.patterns = TRANSFORMATION_PATTERNS
        
        # Categorize seeds
        self.vector_seeds = [s for s in seeds if s.has_vectors]
        self.scalable_vector_seeds = [s for s in seeds if s.has_scalable_vectors]
        self.memory_seeds = [s for s in seeds if s.has_memory_ops]
        self.control_flow_seeds = [s for s in seeds if s.has_control_flow]
        self.call_seeds = [s for s in seeds if s.has_calls]
        
        # Perplexity-based categorization
        self.has_scoring = seed_scores is not None and len(seed_scores) > 0
        if self.has_scoring:
            n_weird = max(1, int(len(seed_scores) * weird_ratio))
            self.weird_seeds = [s for s, _ in seed_scores[:n_weird]]
            self.normal_seeds = [s for s, _ in seed_scores[n_weird:]]
            self.seed_perplexity = {id(s): ppl for s, ppl in seed_scores}
        else:
            self.weird_seeds = []
            self.normal_seeds = []
            self.seed_perplexity = {}
        
        self.prefer_weird_prob = prefer_weird_prob
    
    def _select_seed(
        self, 
        prefer_vectors: bool = False,
        prefer_scalable: bool = False,
        prefer_weird: Optional[bool] = None,
    ) -> tuple[ParsedFunction, bool]:
        """
        Select a seed function based on preferences.
        Returns (seed, is_weird_seed).
        """
        # Decide weird vs normal
        is_weird = False
        if prefer_weird is None and self.has_scoring:
            prefer_weird = random.random() < self.prefer_weird_prob
        
        # Select pool
        if prefer_weird and self.weird_seeds:
            pool = self.weird_seeds
            is_weird = True
        elif prefer_scalable and self.scalable_vector_seeds:
            pool = self.scalable_vector_seeds
        elif prefer_vectors and self.vector_seeds:
            pool = self.vector_seeds
        elif self.has_scoring and self.normal_seeds:
            pool = self.normal_seeds
        else:
            pool = self.seeds
        
        return random.choice(pool), is_weird
    
    def generate_signature_only(
        self,
        prefer_vectors: bool = False,
        prefer_weird: Optional[bool] = None,
    ) -> PromptResult:
        """
        Generate a prompt with just the function signature.
        This is your current best-performing strategy!
        """
        seed, is_weird = self._select_seed(
            prefer_vectors=prefer_vectors,
            prefer_weird=prefer_weird,
        )
        
        new_name = generate_function_name()
        signature = seed.signature.replace(seed.name, new_name)
        
        # Just the signature + opening brace hint
        prompt = f"{signature}\n"
        
        return PromptResult(
            text=prompt,
            strategy=PromptStrategy.SIGNATURE_ONLY,
            metadata={
                "original_name": seed.name,
                "new_name": new_name,
                "is_weird_seed": is_weird,
                "seed_perplexity": self.seed_perplexity.get(id(seed)),
                "seed_has_vectors": seed.has_vectors,
                "seed_has_scalable_vectors": seed.has_scalable_vectors,
            }
        )
    
    def generate_signature_with_entry(
        self,
        prefer_vectors: bool = False,
        prefer_weird: Optional[bool] = None,
    ) -> PromptResult:
        """
        Generate signature + entry block label.
        Gives the model slightly more structure.
        """
        seed, is_weird = self._select_seed(
            prefer_vectors=prefer_vectors,
            prefer_weird=prefer_weird,
        )
        
        new_name = generate_function_name()
        signature = seed.signature.replace(seed.name, new_name)
        
        prompt = f"{signature}\nentry:\n"
        
        return PromptResult(
            text=prompt,
            strategy=PromptStrategy.SIGNATURE_WITH_ENTRY,
            metadata={
                "original_name": seed.name,
                "new_name": new_name,
                "is_weird_seed": is_weird,
                "seed_perplexity": self.seed_perplexity.get(id(seed)),
                "seed_has_vectors": seed.has_vectors,
                "seed_has_scalable_vectors": seed.has_scalable_vectors,
            }
        )
    
    def generate_pattern_guided(
        self,
        prefer_vectors: bool = False,
        prefer_weird: Optional[bool] = None,
        pattern: Optional[TransformationPattern] = None,
    ) -> PromptResult:
        """
        Generate a prompt with a semantic hint about what to generate.
        """
        seed, is_weird = self._select_seed(
            prefer_vectors=prefer_vectors,
            prefer_weird=prefer_weird,
        )
        
        if pattern is None:
            pattern = random.choice(self.patterns)
        
        new_name = generate_function_name()
        signature = seed.signature.replace(seed.name, new_name)
        
        # Create a semantic hint comment
        hint = random.choice(pattern.input_hints)
        
        prompt = f"; {pattern.description}\n; Contains: {hint}\n{signature}\n"
        
        return PromptResult(
            text=prompt,
            strategy=PromptStrategy.PATTERN_GUIDED,
            metadata={
                "original_name": seed.name,
                "new_name": new_name,
                "pattern_name": pattern.name,
                "is_weird_seed": is_weird,
                "seed_perplexity": self.seed_perplexity.get(id(seed)),
                "seed_has_vectors": seed.has_vectors,
                "seed_has_scalable_vectors": seed.has_scalable_vectors,
            }
        )
    
    def generate_type_novel(
        self,
        prefer_weird: Optional[bool] = None,
    ) -> PromptResult:
        """
        Take a seed signature and substitute with unusual/interesting types.
        Focus on scalable vectors and unusual integer widths.
        """
        # Prefer seeds with vectors so substitution makes sense
        seed, is_weird = self._select_seed(
            prefer_vectors=True,
            prefer_scalable=True,
            prefer_weird=prefer_weird,
        )
        
        new_name = generate_function_name()
        signature = seed.signature.replace(seed.name, new_name)
        
        # Apply type substitutions
        modified_sig = substitute_types_in_signature(signature, [])
        
        prompt = f"{modified_sig}\n"
        
        return PromptResult(
            text=prompt,
            strategy=PromptStrategy.TYPE_NOVEL,
            metadata={
                "original_name": seed.name,
                "new_name": new_name,
                "original_signature": seed.signature,
                "is_weird_seed": is_weird,
                "seed_perplexity": self.seed_perplexity.get(id(seed)),
                "seed_has_vectors": seed.has_vectors,
                "seed_has_scalable_vectors": seed.has_scalable_vectors,
                "type_modified": True,
            }
        )
    
    def generate_truncated_clean(
        self,
        prefer_vectors: bool = False,
        prefer_weird: Optional[bool] = None,
        min_ratio: float = 0.3,
        max_ratio: float = 0.6,
    ) -> PromptResult:
        """
        Old-style truncation but with FileCheck comments stripped.
        For comparison with the cleaner approaches.
        """
        seed, is_weird = self._select_seed(
            prefer_vectors=prefer_vectors,
            prefer_weird=prefer_weird,
        )
        
        new_name = generate_function_name()
        
        # Get clean version (no FileCheck)
        clean_sig = seed.signature.replace(seed.name, new_name)
        clean_body = extract_clean_body(seed)
        
        full_clean = clean_sig + '\n' + '\n'.join(clean_body)
        
        # Truncate
        ratio = random.uniform(min_ratio, max_ratio)
        cut_point = int(len(full_clean) * ratio)
        
        # Try to cut at newline
        newline_pos = full_clean.rfind('\n', 0, cut_point)
        if newline_pos > len(clean_sig):
            prompt = full_clean[:newline_pos + 1]
        else:
            prompt = full_clean[:cut_point]
        
        return PromptResult(
            text=prompt,
            strategy=PromptStrategy.TRUNCATED_FULL,
            metadata={
                "original_name": seed.name,
                "new_name": new_name,
                "is_weird_seed": is_weird,
                "seed_perplexity": self.seed_perplexity.get(id(seed)),
                "seed_has_vectors": seed.has_vectors,
                "seed_has_scalable_vectors": seed.has_scalable_vectors,
                "truncate_ratio": ratio,
            }
        )
    
    def generate(
        self,
        strategy: Optional[PromptStrategy] = None,
        prefer_vectors: bool = False,
        prefer_weird: Optional[bool] = None,
    ) -> PromptResult:
        """
        Generate a prompt using the specified or random strategy.
        """
        if strategy is None:
            # Weight strategies - favor the ones that work
            weights = [
                (PromptStrategy.SIGNATURE_ONLY, 4),      # Your best performer
                (PromptStrategy.SIGNATURE_WITH_ENTRY, 3),
                (PromptStrategy.PATTERN_GUIDED, 2),
                (PromptStrategy.TYPE_NOVEL, 3),          # Good for finding edge cases
                (PromptStrategy.TRUNCATED_FULL, 1),      # Baseline comparison
            ]
            strategies, probs = zip(*weights)
            strategy = random.choices(strategies, weights=probs)[0]
        
        if strategy == PromptStrategy.SIGNATURE_ONLY:
            return self.generate_signature_only(prefer_vectors, prefer_weird)
        elif strategy == PromptStrategy.SIGNATURE_WITH_ENTRY:
            return self.generate_signature_with_entry(prefer_vectors, prefer_weird)
        elif strategy == PromptStrategy.PATTERN_GUIDED:
            return self.generate_pattern_guided(prefer_vectors, prefer_weird)
        elif strategy == PromptStrategy.TYPE_NOVEL:
            return self.generate_type_novel(prefer_weird)
        elif strategy == PromptStrategy.TRUNCATED_FULL:
            return self.generate_truncated_clean(prefer_vectors, prefer_weird)
        else:
            return self.generate_signature_only(prefer_vectors, prefer_weird)
    
    def generate_batch(
        self,
        batch_size: int,
        vector_ratio: float = 0.3,
        scalable_ratio: float = 0.2,
        strategy: Optional[PromptStrategy] = None,
    ) -> list[tuple[str, dict]]:
        """
        Generate a batch of prompts.
        
        Returns list of (prompt_text, metadata) tuples.
        """
        results = []
        n_vector = int(batch_size * vector_ratio)
        n_scalable = int(batch_size * scalable_ratio)
        
        for i in range(batch_size):
            # Determine preferences for this sample
            if i < n_scalable:
                # Force scalable vectors via type_novel
                result = self.generate(
                    strategy=PromptStrategy.TYPE_NOVEL,
                    prefer_vectors=True,
                )
            elif i < n_vector:
                result = self.generate(
                    strategy=strategy,
                    prefer_vectors=True,
                )
            else:
                result = self.generate(strategy=strategy)
            
            # Add strategy to metadata
            result.metadata["strategy"] = result.strategy.value
            results.append((result.text, result.metadata))
        
        return results
    
    def stats(self) -> dict:
        """Return statistics about the generator."""
        return {
            "total_seeds": len(self.seeds),
            "vector_seeds": len(self.vector_seeds),
            "scalable_vector_seeds": len(self.scalable_vector_seeds),
            "memory_seeds": len(self.memory_seeds),
            "control_flow_seeds": len(self.control_flow_seeds),
            "call_seeds": len(self.call_seeds),
            "has_scoring": self.has_scoring,
            "weird_seeds": len(self.weird_seeds),
            "normal_seeds": len(self.normal_seeds),
            "num_patterns": len(self.patterns),
        }


# Convenience function
def collect_seeds(
    test_dir: Path,
    max_count: int = 10000,
    min_length: int = 100,
    max_length: int = 2000,
) -> list[ParsedFunction]:
    """Collect seed functions from LLVM test directory."""
    all_functions = parse_functions_from_directory(test_dir, max_count=max_count * 2)
    
    filtered = [
        f for f in all_functions
        if min_length < len(f.raw_text) < max_length
    ]
    
    return filtered[:max_count]