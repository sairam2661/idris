"""
Diverse LLVM IR Generator for Fuzzing

Uses LLM completion with various strategies to generate diverse test cases.
Leverages the full LLVM test corpus for context, types, and patterns.
"""

import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from enum import Enum



class Strategy(Enum):
    """Generation strategies - each produces different kinds of IR."""
    TYPE_SHIFTED = "type_shifted"           # Force unusual types
    FULL_CONTEXT = "full_context"           # Include declarations, target
    VARIATION = "variation"                 # Show example, ask for different
    CROSS_POLLINATE = "cross_pollinate"     # Mix structure + types
    LOOP_FOCUSED = "loop_focused"           # Generate loop structures
    MEMORY_FOCUSED = "memory_focused"       # Generate load/store patterns
    VECTOR_OPS = "vector_ops"               # Vector operation patterns
    INTRINSIC_HEAVY = "intrinsic_heavy"     # Use LLVM intrinsics


# Interesting types that often expose bugs
SPICY_TYPES = {
    "scalable_vectors": [
        "<vscale x 1 x i1>",
        "<vscale x 2 x i1>",
        "<vscale x 4 x i1>",
        "<vscale x 8 x i1>",
        "<vscale x 2 x i8>",
        "<vscale x 4 x i8>",
        "<vscale x 8 x i8>",
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
        "<16 x i1>",
        "<2 x i8>",
        "<4 x i8>",
        "<16 x i8>",
        "<32 x i8>",
        "<2 x i16>",
        "<4 x i16>",
        "<8 x i16>",
        "<2 x i32>",
        "<4 x i32>",
        "<8 x i32>",
        "<3 x i32>",   # Non-power-of-2
        "<5 x i32>",
        "<7 x i32>",
        "<2 x i64>",
        "<4 x i64>",
        "<2 x i128>",
        "<4 x float>",
        "<8 x float>",
        "<2 x double>",
        "<4 x double>",
        "<4 x half>",
        "<8 x half>",
        "<4 x bfloat>",
    ],
    "unusual_integers": [
        "i1",
        "i3",
        "i5", 
        "i7",
        "i17",
        "i33",
        "i48",
        "i63",
        "i65",
        "i128",
        "i256",
    ],
    "floats": [
        "half",
        "bfloat",
        "float",
        "double",
        "fp128",
    ],
}

# All spicy types flattened
ALL_SPICY_TYPES = (
    SPICY_TYPES["scalable_vectors"] + 
    SPICY_TYPES["fixed_vectors"] + 
    SPICY_TYPES["unusual_integers"] +
    SPICY_TYPES["floats"]
)

# Common LLVM intrinsics worth testing
INTRINSICS = {
    "math": [
        "llvm.abs.{T}",
        "llvm.smax.{T}",
        "llvm.smin.{T}",
        "llvm.umax.{T}",
        "llvm.umin.{T}",
        "llvm.ctpop.{T}",
        "llvm.ctlz.{T}",
        "llvm.cttz.{T}",
        "llvm.bitreverse.{T}",
        "llvm.bswap.{T}",
        "llvm.fshl.{T}",
        "llvm.fshr.{T}",
    ],
    "overflow": [
        "llvm.sadd.with.overflow.{T}",
        "llvm.uadd.with.overflow.{T}",
        "llvm.ssub.with.overflow.{T}",
        "llvm.usub.with.overflow.{T}",
        "llvm.smul.with.overflow.{T}",
        "llvm.umul.with.overflow.{T}",
    ],
    "saturating": [
        "llvm.sadd.sat.{T}",
        "llvm.uadd.sat.{T}",
        "llvm.ssub.sat.{T}",
        "llvm.usub.sat.{T}",
    ],
    "reduction": [
        "llvm.vector.reduce.add.{T}",
        "llvm.vector.reduce.mul.{T}",
        "llvm.vector.reduce.and.{T}",
        "llvm.vector.reduce.or.{T}",
        "llvm.vector.reduce.xor.{T}",
        "llvm.vector.reduce.smax.{T}",
        "llvm.vector.reduce.smin.{T}",
        "llvm.vector.reduce.umax.{T}",
        "llvm.vector.reduce.umin.{T}",
    ],
    "float_math": [
        "llvm.fabs.{T}",
        "llvm.sqrt.{T}",
        "llvm.fma.{T}",
        "llvm.fmuladd.{T}",
        "llvm.maxnum.{T}",
        "llvm.minnum.{T}",
        "llvm.maximum.{T}",
        "llvm.minimum.{T}",
        "llvm.copysign.{T}",
        "llvm.floor.{T}",
        "llvm.ceil.{T}",
        "llvm.trunc.{T}",
        "llvm.rint.{T}",
        "llvm.round.{T}",
    ],
    "vector": [
        "llvm.vector.insert.{VT}",
        "llvm.vector.extract.{VT}",
        "llvm.vector.splice.{VT}",
        "llvm.vector.reverse.{VT}",
    ],
    "masked": [
        "llvm.masked.load.{VT}",
        "llvm.masked.store.{VT}",
        "llvm.masked.gather.{VT}",
        "llvm.masked.scatter.{VT}",
    ],
}

# Target triples to use
TARGET_TRIPLES = [
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "x86_64-apple-macosx",
    "aarch64-apple-macosx",
]


@dataclass
class ParsedTestFile:
    """Structured representation of an LLVM test file."""
    path: str
    pass_name: str
    target_triple: str
    functions: list  # list of ParsedFunction
    declarations: list[str]
    attributes: list[str]
    
    @property
    def has_vectors(self) -> bool:
        return any(f.has_vectors for f in self.functions)
    
    @property
    def has_scalable_vectors(self) -> bool:
        return any(f.has_scalable_vectors for f in self.functions)
    
    @property
    def has_loops(self) -> bool:
        return any(getattr(f, 'has_control_flow', False) for f in self.functions)


def generate_name() -> str:
    """
    Generate function name.
    
    Using @test consistently allows us to deduplicate by signature pattern.
    The actual uniqueness comes from file paths, not function names.
    """
    return "@test"

def get_scalar_type(vector_type: str) -> str:
    """Extract scalar type from vector type."""
    # "<4 x i32>" -> "i32"
    # "<vscale x 2 x float>" -> "float"
    match = re.search(r'x\s+(\w+)>$', vector_type)
    if match:
        return match.group(1)
    return "i32"


def is_vector_type(t: str) -> bool:
    """Check if type is a vector."""
    return t.startswith("<") and "x" in t


class SemanticAxis(Enum):
    POISON = "poison"
    ALIASING = "aliasing"
    CONTROL = "control"
        

def is_float_type(t: str) -> bool:
    """Check if type is floating point."""
    return t in ["half", "bfloat", "float", "double", "fp128"] or \
           any(ft in t for ft in ["half", "bfloat", "float", "double", "fp128"])


def is_integer_type(t: str) -> bool:
    """Check if type is integer."""
    if is_float_type(t):
        return False
    if "ptr" in t:
        return False
    return True


class DiverseGenerator:
    """
    Generates diverse LLVM IR test cases using various LLM prompting strategies.
    """
    
    def __init__(self, corpus: list[ParsedTestFile], track_signatures: bool = True):
        """
        Args:
            corpus: Parsed LLVM test files
            track_signatures: If True, avoid generating duplicate signatures
        """
        self.corpus = corpus
        self.track_signatures = track_signatures
        self.seen_signatures: set[str] = set()
        
        # Index by pass for stratified sampling
        self.by_pass = {}
        for test_file in corpus:
            pass_name = test_file.pass_name or "unknown"
            if pass_name not in self.by_pass:
                self.by_pass[pass_name] = []
            self.by_pass[pass_name].append(test_file)
        
        # Index by features
        self.with_vectors = [t for t in corpus if t.has_vectors]
        self.with_scalable = [t for t in corpus if t.has_scalable_vectors]
        self.with_loops = [t for t in corpus if t.has_loops]
        
        # Strategy weights (favor what works)
        self.strategy_weights = {
            Strategy.TYPE_SHIFTED: 0,       # High yield
            Strategy.FULL_CONTEXT: 10,       
            Strategy.VARIATION: 0,
            Strategy.CROSS_POLLINATE: 0,
            Strategy.LOOP_FOCUSED: 0,
            Strategy.MEMORY_FOCUSED: 0,
            Strategy.VECTOR_OPS: 0,
            Strategy.INTRINSIC_HEAVY: 0,
        }
   
    def adversarial_preamble(self) -> str:
        return (
            "; Fuzzing-oriented LLVM IR\n"
            "; This IR must be valid but intentionally adversarial\n"
            "; Stress optimizer edge cases and incorrect assumptions\n"
            "; Prefer minimal and awkward constructions over idiomatic IR\n\n"
            ""
        )
     
    def _normalize_signature(self, prompt: str) -> str:
        """
        Extract normalized signature for deduplication.
        
        E.g., 'define <2 x i1> @test(<2 x i1> %a)' -> '<2 x i1>(<2 x i1>)'
        """
        # Find the define line
        for line in prompt.split('\n'):
            if line.strip().startswith('define '):
                # Extract return type and arg types
                # Remove function name and attributes
                match = re.match(r'define\s+(.+?)\s+@\w+\(([^)]*)\)', line)
                if match:
                    ret_type = match.group(1).strip()
                    args = match.group(2).strip()
                    # Normalize arg names
                    args = re.sub(r'%\w+', '%x', args)
                    return f"{ret_type}({args})"
        return prompt[:100]  # Fallback
    
    def _is_duplicate_signature(self, prompt: str) -> bool:
        """Check if we've already generated this signature pattern."""
        if not self.track_signatures:
            return False
        sig = self._normalize_signature(prompt)
        if sig in self.seen_signatures:
            return True
        self.seen_signatures.add(sig)
        return False
    
    def sample_test_file(self, prefer_vectors: bool = False, 
                         prefer_loops: bool = False) -> ParsedTestFile:
        """Sample a test file, stratified by pass."""
        
        if prefer_vectors and self.with_scalable and random.random() < 0.5:
            return random.choice(self.with_scalable)
        if prefer_vectors and self.with_vectors and random.random() < 0.5:
            return random.choice(self.with_vectors)
        if prefer_loops and self.with_loops and random.random() < 0.5:
            return random.choice(self.with_loops)
        
        # Stratified by pass
        pass_name = random.choice(list(self.by_pass.keys()))
        return random.choice(self.by_pass[pass_name])
    
    def pick_strategy(self) -> Strategy:
        """Pick a generation strategy based on weights."""
        strategies = list(self.strategy_weights.keys())
        weights = list(self.strategy_weights.values())
        return random.choices(strategies, weights=weights)[0]
    
    def pick_spicy_type(self, category: Optional[str] = None) -> str:
        """Pick an interesting type."""
        if category and category in SPICY_TYPES:
            return random.choice(SPICY_TYPES[category])
        return random.choice(ALL_SPICY_TYPES)
    
    def pick_target_triple(self) -> str:
        """Pick a target triple."""
        return random.choice(TARGET_TRIPLES)
    
    def generate(self, max_retries: int = 5) -> tuple[str, dict]:
        """
        Generate one test case.
        
        Args:
            max_retries: Max attempts to generate a unique signature
        
        Returns:
            (prompt, metadata) - prompt ready for LLM completion
        """
        for _ in range(max_retries):
            strategy = self.pick_strategy()
            
            if strategy == Strategy.TYPE_SHIFTED:
                prompt, meta = self.prompt_type_shifted()
            elif strategy == Strategy.FULL_CONTEXT:
                prompt, meta = self.prompt_full_context()
            elif strategy == Strategy.VARIATION:
                prompt, meta = self.prompt_variation()
            elif strategy == Strategy.CROSS_POLLINATE:
                prompt, meta = self.prompt_cross_pollinate()
            elif strategy == Strategy.LOOP_FOCUSED:
                prompt, meta = self.prompt_loop_focused()
            elif strategy == Strategy.MEMORY_FOCUSED:
                prompt, meta = self.prompt_memory_focused()
            elif strategy == Strategy.VECTOR_OPS:
                prompt, meta = self.prompt_vector_ops()
            elif strategy == Strategy.INTRINSIC_HEAVY:
                prompt, meta = self.prompt_intrinsic_heavy()
            else:
                prompt, meta = self.prompt_type_shifted()
            
            # Check for duplicate signature
            if not self._is_duplicate_signature(prompt):
                return prompt, meta
        
        # If we exhausted retries, return anyway (better than nothing)
        return prompt, meta
    
    def prompt_type_shifted(self) -> tuple[str, dict]:
        """Force interesting types into signature."""
        
        test_file = self.sample_test_file(prefer_vectors=True)
        spicy = self.pick_spicy_type()
        target = test_file.target_triple or self.pick_target_triple()
        
        # For scalable vectors, prefer aarch64
        if "vscale" in spicy:
            target = "aarch64-unknown-linux-gnu"
        
        name = generate_name()
        n_args = random.randint(1, 3)
        args = ", ".join([f"{spicy} %arg{i}" for i in range(n_args)])
        
        prompt = self.adversarial_preamble()
        prompt += f'target triple = "{target}"\n\n'
        prompt += f"define {spicy} {name}({args}) {{\n"
        
        # Maybe add entry label
        if random.random() < 0.5:
            prompt += "entry:\n"
        
        return prompt, {
            "strategy": Strategy.TYPE_SHIFTED.value,
            "injected_type": spicy,
            "source_pass": test_file.pass_name,
            "target_triple": target,
        }
    
    def prompt_full_context(self) -> tuple[str, dict]:
        """Include declarations and target from test file."""
        
        test_file = self.sample_test_file()
        target = test_file.target_triple or self.pick_target_triple()
        
        if not test_file.functions:
            return self.prompt_type_shifted()  # fallback
        
        func = random.choice(test_file.functions)
        name = generate_name()
        
        # Replace function name in signature
        new_sig = func.signature.replace(func.name, name)
        
        prompt = self.adversarial_preamble()
        prompt += f'target triple = "{target}"\n\n'
        
        # Add some declarations
        for decl in test_file.declarations[:6]:
            prompt += decl + "\n"
        
        if test_file.declarations:
            prompt += "\n"
        
        prompt += f"{new_sig}\n"
        
        return prompt, {
            "strategy": Strategy.FULL_CONTEXT.value,
            "source_pass": test_file.pass_name,
            "source_file": test_file.path,
            "target_triple": target,
        }
    
    def prompt_variation(self) -> tuple[str, dict]:
        """Show example, ask for variation."""
        
        test_file = self.sample_test_file()
        target = test_file.target_triple or self.pick_target_triple()
        
        if not test_file.functions:
            return self.prompt_type_shifted()
        
        func = random.choice(test_file.functions)
        name = generate_name()
        
        prompt = self.adversarial_preamble()
        prompt += f'target triple = "{target}"\n\n'
        prompt += f"; Example function:\n"
        prompt += func.signature + "\n"
        
        # Include body if not too long
        body = ""
        for line in func.body_lines:
            if not line.strip().startswith(';'):  # Skip comments
                body += line + "\n"
        
        body_lines = body.strip().split('\n')
        if len(body_lines) <= 20:
            prompt += body + "\n"
        else:
            prompt += '\n'.join(body_lines[:10]) + "\n  ; ...\n"
        
        prompt += "}\n\n"
        prompt += "; Different function with similar structure:\n"
        
        # New signature, possibly with different type
        if random.random() < 0.5:
            spicy = self.pick_spicy_type()
            n_args = random.randint(1, 3)
            args = ", ".join([f"{spicy} %arg{i}" for i in range(n_args)])
            prompt += f"define {spicy} {name}({args}) {{\n"
        else:
            new_sig = func.signature.replace(func.name, name)
            prompt += f"{new_sig}\n"
        
        return prompt, {
            "strategy": Strategy.VARIATION.value,
            "source_pass": test_file.pass_name,
            "example_function": func.name,
            "target_triple": target,
        }
    
    def prompt_cross_pollinate(self) -> tuple[str, dict]:
        """Mix structure from one test, types from another."""
        
        # Get a test file for structure
        struct_file = self.sample_test_file(prefer_loops=True)
        target = struct_file.target_triple or self.pick_target_triple()
        
        if not struct_file.functions:
            return self.prompt_type_shifted()
        
        struct_func = random.choice(struct_file.functions)
        
        # Pick a spicy type
        spicy = self.pick_spicy_type()
        if "vscale" in spicy:
            target = "aarch64-unknown-linux-gnu"
        
        name = generate_name()
        n_args = random.randint(1, 3)
        args = ", ".join([f"{spicy} %arg{i}" for i in range(n_args)])
        
        prompt = self.adversarial_preamble()
        prompt += f'target triple = "{target}"\n\n'
        prompt += f"define {spicy} {name}({args}) {{\n"
        
        # Copy some structure from original (labels, branches)
        kept = 0
        for line in struct_func.body_lines[:5]:
            line_stripped = line.strip()
            # Skip comments
            if line_stripped.startswith(';'):
                continue
            # Keep labels
            if line_stripped.endswith(':') and not line_stripped.startswith(';'):
                prompt += line + "\n"
                kept += 1
            # Keep branches
            elif line_stripped.startswith('br '):
                prompt += line + "\n"
                kept += 1
                break
        
        if kept == 0:
            prompt += "entry:\n"
        
        return prompt, {
            "strategy": Strategy.CROSS_POLLINATE.value,
            "source_pass": struct_file.pass_name,
            "structure_from": struct_func.name,
            "injected_type": spicy,
            "target_triple": target,
        }
    
    def prompt_loop_focused(self) -> tuple[str, dict]:
        """Generate loop structures."""
        
        test_file = self.sample_test_file(prefer_loops=True)
        target = test_file.target_triple or self.pick_target_triple()
        
        spicy = self.pick_spicy_type()
        if "vscale" in spicy:
            target = "aarch64-unknown-linux-gnu"
        
        name = generate_name()
        
        # Different loop patterns
        loop_pattern = random.choice(["simple", "nested", "reduction", "with_exit"])
        
        prompt = self.adversarial_preamble()
        prompt += f'target triple = "{target}"\n\n'
        
        if loop_pattern == "simple":
            prompt += f"define {spicy} {name}({spicy} %init, i32 %n) {{\n"
            prompt += "entry:\n"
            prompt += "  br label %loop\n"
            prompt += "loop:\n"
            prompt += f"  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]\n"
            prompt += f"  %acc = phi {spicy} [ %init, %entry ], [ %acc.next, %loop ]\n"
        
        elif loop_pattern == "nested":
            prompt += f"define {spicy} {name}({spicy} %init, i32 %n, i32 %m) {{\n"
            prompt += "entry:\n"
            prompt += "  br label %outer\n"
            prompt += "outer:\n"
            prompt += f"  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]\n"
            prompt += "  br label %inner\n"
            prompt += "inner:\n"
            prompt += f"  %j = phi i32 [ 0, %outer ], [ %j.next, %inner ]\n"
        
        elif loop_pattern == "reduction":
            scalar = get_scalar_type(spicy) if is_vector_type(spicy) else spicy
            prompt += f"define {scalar} {name}({spicy} %vec, i32 %n) {{\n"
            prompt += "entry:\n"
            prompt += "  br label %loop\n"
            prompt += "loop:\n"
            prompt += f"  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]\n"
        
        else:  # with_exit
            prompt += f"define {spicy} {name}({spicy} %init, i32 %n, i1 %cond) {{\n"
            prompt += "entry:\n"
            prompt += "  br label %loop\n"
            prompt += "loop:\n"
            prompt += f"  %i = phi i32 [ 0, %entry ], [ %i.next, %continue ]\n"
            prompt += f"  %val = phi {spicy} [ %init, %entry ], [ %val.next, %continue ]\n"
            prompt += "  br i1 %cond, label %early_exit, label %continue\n"
            prompt += "continue:\n"
        
        return prompt, {
            "strategy": Strategy.LOOP_FOCUSED.value,
            "loop_pattern": loop_pattern,
            "injected_type": spicy,
            "source_pass": test_file.pass_name,
            "target_triple": target,
        }
    
    def prompt_memory_focused(self) -> tuple[str, dict]:
        """Generate memory access patterns."""
        
        test_file = self.sample_test_file()
        target = test_file.target_triple or self.pick_target_triple()
        
        spicy = self.pick_spicy_type()
        if "vscale" in spicy:
            target = "aarch64-unknown-linux-gnu"
        
        name = generate_name()
        
        # Different memory patterns
        mem_pattern = random.choice(["load_store", "gep", "memcpy_pattern", "aliasing"])
        
        prompt = self.adversarial_preamble()
        prompt += f'target triple = "{target}"\n\n'
        
        if mem_pattern == "load_store":
            prompt += f"define {spicy} {name}(ptr %p, ptr %q, {spicy} %val) {{\n"
            prompt += "entry:\n"
            prompt += f"  store {spicy} %val, ptr %p\n"
            prompt += f"  %v1 = load {spicy}, ptr %p\n"
        
        elif mem_pattern == "gep":
            prompt += f"define {spicy} {name}(ptr %base, i64 %idx) {{\n"
            prompt += "entry:\n"
            prompt += f"  %ptr = getelementptr {spicy}, ptr %base, i64 %idx\n"
            prompt += f"  %val = load {spicy}, ptr %ptr\n"
        
        elif mem_pattern == "memcpy_pattern":
            prompt += f"define void {name}(ptr %dst, ptr %src, i64 %n) {{\n"
            prompt += "entry:\n"
            prompt += "  br label %loop\n"
            prompt += "loop:\n"
            prompt += "  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]\n"
            prompt += f"  %src.ptr = getelementptr {spicy}, ptr %src, i64 %i\n"
            prompt += f"  %dst.ptr = getelementptr {spicy}, ptr %dst, i64 %i\n"
        
        else:  # aliasing
            prompt += f"define {spicy} {name}(ptr %p, ptr %q, {spicy} %v1, {spicy} %v2) {{\n"
            prompt += "entry:\n"
            prompt += f"  store {spicy} %v1, ptr %p\n"
            prompt += f"  store {spicy} %v2, ptr %q\n"
            prompt += f"  %r1 = load {spicy}, ptr %p\n"
        
        return prompt, {
            "strategy": Strategy.MEMORY_FOCUSED.value,
            "memory_pattern": mem_pattern,
            "injected_type": spicy,
            "source_pass": test_file.pass_name,
            "target_triple": target,
        }
    
    def prompt_vector_ops(self) -> tuple[str, dict]:
        """Generate vector operation patterns."""
        
        test_file = self.sample_test_file(prefer_vectors=True)
        
        # Pick a vector type
        vec_category = random.choice(["scalable_vectors", "fixed_vectors"])
        spicy = self.pick_spicy_type(vec_category)
        
        # Scalable vectors need aarch64
        if "vscale" in spicy:
            target = "aarch64-unknown-linux-gnu"
        else:
            target = test_file.target_triple or self.pick_target_triple()
        
        name = generate_name()
        scalar = get_scalar_type(spicy)
        
        # Different vector patterns
        vec_pattern = random.choice([
            "shuffle", "extract_insert", "splat", "binop", "compare", "select"
        ])
        
        prompt = self.adversarial_preamble()
        prompt += f'target triple = "{target}"\n\n'
        
        if vec_pattern == "shuffle":
            prompt += f"define {spicy} {name}({spicy} %a, {spicy} %b) {{\n"
            prompt += "entry:\n"
            prompt += f"  %shuf = shufflevector {spicy} %a, {spicy} %b, "
        
        elif vec_pattern == "extract_insert":
            prompt += f"define {spicy} {name}({spicy} %vec, {scalar} %val, i32 %idx) {{\n"
            prompt += "entry:\n"
            prompt += f"  %e = extractelement {spicy} %vec, i32 %idx\n"
        
        elif vec_pattern == "splat":
            prompt += f"define {spicy} {name}({scalar} %val) {{\n"
            prompt += "entry:\n"
            # Model should figure out splat pattern
        
        elif vec_pattern == "binop":
            prompt += f"define {spicy} {name}({spicy} %a, {spicy} %b) {{\n"
            prompt += "entry:\n"
            op = random.choice(["add", "sub", "mul", "and", "or", "xor"] if is_integer_type(scalar) 
                              else ["fadd", "fsub", "fmul", "fdiv"])
            prompt += f"  %r1 = {op} {spicy} %a, %b\n"
        
        elif vec_pattern == "compare":
            prompt += f"define {spicy} {name}({spicy} %a, {spicy} %b) {{\n"
            prompt += "entry:\n"
            if is_integer_type(scalar):
                pred = random.choice(["eq", "ne", "slt", "sgt", "sle", "sge", "ult", "ugt"])
                prompt += f"  %cmp = icmp {pred} {spicy} %a, %b\n"
            else:
                pred = random.choice(["oeq", "one", "olt", "ogt", "ole", "oge"])
                prompt += f"  %cmp = fcmp {pred} {spicy} %a, %b\n"
        
        else:  # select
            # Simplified - let model figure out the mask type
            prompt += f"define {spicy} {name}({spicy} %a, {spicy} %b) {{\n"
            prompt += "entry:\n"
            if is_integer_type(scalar):
                prompt += f"  %cmp = icmp sgt {spicy} %a, %b\n"
            else:
                prompt += f"  %cmp = fcmp ogt {spicy} %a, %b\n"
            prompt += f"  %sel = select "
        
        return prompt, {
            "strategy": Strategy.VECTOR_OPS.value,
            "vector_pattern": vec_pattern,
            "injected_type": spicy,
            "source_pass": test_file.pass_name,
            "target_triple": target,
        }
    
    def prompt_intrinsic_heavy(self) -> tuple[str, dict]:
        """Generate code using LLVM intrinsics."""
        
        test_file = self.sample_test_file()
        target = test_file.target_triple or self.pick_target_triple()
        
        # Pick a type and matching intrinsics
        type_choice = random.choice(["integer", "float", "vector"])
        
        if type_choice == "integer":
            spicy = random.choice(["i32", "i64", "i128", "i17"])
            intrinsic_pool = INTRINSICS["math"] + INTRINSICS["overflow"] + INTRINSICS["saturating"]
        elif type_choice == "float":
            spicy = random.choice(["float", "double", "half"])
            intrinsic_pool = INTRINSICS["float_math"]
        else:
            spicy = self.pick_spicy_type("fixed_vectors")
            if is_float_type(get_scalar_type(spicy)):
                intrinsic_pool = INTRINSICS["float_math"] + INTRINSICS["reduction"]
            else:
                intrinsic_pool = INTRINSICS["math"] + INTRINSICS["reduction"]
        
        name = generate_name()
        
        prompt = self.adversarial_preamble()
        prompt += f'target triple = "{target}"\n\n'
        
        # Pick some intrinsics
        selected = random.sample(intrinsic_pool, min(3, len(intrinsic_pool)))
        
        # Declare them
        for intr in selected:
            intr_name = intr.replace("{T}", spicy).replace("{VT}", spicy)
            # Build declaration (simplified)
            if "overflow" in intr_name:
                prompt += f"declare {{ {spicy}, i1 }} @{intr_name}({spicy}, {spicy})\n"
            elif "reduce" in intr_name:
                scalar = get_scalar_type(spicy)
                prompt += f"declare {scalar} @{intr_name}({spicy})\n"
            else:
                prompt += f"declare {spicy} @{intr_name}({spicy})\n"
        
        prompt += f"\ndefine {spicy} {name}({spicy} %a, {spicy} %b) {{\n"
        prompt += "entry:\n"
        
        # Start with one intrinsic call
        intr = random.choice(selected)
        intr_name = intr.replace("{T}", spicy).replace("{VT}", spicy)
        
        if "overflow" in intr_name:
            prompt += f"  %res = call {{ {spicy}, i1 }} @{intr_name}({spicy} %a, {spicy} %b)\n"
            prompt += f"  %val = extractvalue {{ {spicy}, i1 }} %res, 0\n"
        elif "reduce" in intr_name:
            prompt += f"  %r = call {get_scalar_type(spicy)} @{intr_name}({spicy} %a)\n"
        else:
            prompt += f"  %r1 = call {spicy} @{intr_name}({spicy} %a)\n"
        
        return prompt, {
            "strategy": Strategy.INTRINSIC_HEAVY.value,
            "intrinsics": [i.replace("{T}", spicy).replace("{VT}", spicy) for i in selected],
            "injected_type": spicy,
            "source_pass": test_file.pass_name,
            "target_triple": target,
        }
    
    def generate_batch(self, batch_size: int) -> list[tuple[str, dict]]:
        """Generate a batch of prompts."""
        return [self.generate() for _ in range(batch_size)]
    
    def stats(self) -> dict:
        """Return corpus statistics."""
        return {
            "total_test_files": len(self.corpus),
            "passes": list(self.by_pass.keys()),
            "num_passes": len(self.by_pass),
            "with_vectors": len(self.with_vectors),
            "with_scalable": len(self.with_scalable),
            "with_loops": len(self.with_loops),
            "unique_signatures_generated": len(self.seen_signatures),
            "tracking_signatures": self.track_signatures,
        }
    
    def reset_signature_tracking(self):
        """Clear seen signatures to allow regenerating patterns."""
        self.seen_signatures.clear()

    def semantic_comment(self, axis: SemanticAxis) -> str:
        if axis == SemanticAxis.POISON:
            return "; Introduce poison values that are partially masked\n"
        if axis == SemanticAxis.ALIASING:
            return "; Pointer aliasing relationships should be unclear\n"
        if axis == SemanticAxis.CONTROL:
            return "; Control flow should inhibit straightforward simplification\n"
        return ""


def build_corpus(test_dir: Path, max_files: int = 20000) -> list[ParsedTestFile]:
    """
    Build corpus from LLVM test directory.
    
    Uses the parser module to extract structured information.
    """
    from idris.parser import parse_functions_from_file
    
    corpus = []
    
    for ll_file in test_dir.rglob("*.ll"):
        if len(corpus) >= max_files:
            break
        
        try:
            content = ll_file.read_text()
        except:
            continue
        
        # Extract pass name from RUN line
        pass_match = re.search(r'-passes=([^\s|]+)', content)
        pass_name = pass_match.group(1) if pass_match else "unknown"
        # Clean up pass name
        pass_name = pass_name.split(',')[0].strip("'\"")
        if '<' in pass_name:
            pass_name = pass_name.split('<')[0]
        
        # Extract target triple
        target_match = re.search(r'target triple = "([^"]+)"', content)
        target_triple = target_match.group(1) if target_match else ""
        
        # Extract declarations
        declarations = re.findall(r'^declare [^\n]+', content, re.MULTILINE)
        
        # Extract attributes
        attributes = re.findall(r'^attributes #\d+ = \{[^}]+\}', content, re.MULTILINE)
        
        # Parse functions
        functions = parse_functions_from_file(ll_file)
        
        if functions:
            corpus.append(ParsedTestFile(
                path=str(ll_file),
                pass_name=pass_name,
                target_triple=target_triple,
                functions=functions,
                declarations=declarations,
                attributes=attributes,
            ))
    
    return corpus