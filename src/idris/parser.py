import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class CheckDirective:
    """A single FileCheck directive line"""
    prefix: str          # e.g., "VF2", "CHECK", "AVX2"
    kind: str            # e.g., "LABEL", "NEXT", "SAME", "NOT", "DAG"
    content: str         # The pattern/text after the directive
    raw_line: str        # Original line
    
    
@dataclass 
class CheckBlock:
    """A group of related CHECK directives (usually for one output section)"""
    directives: list[CheckDirective] = field(default_factory=list)
    
    def to_text(self) -> str:
        return '\n'.join(d.raw_line for d in self.directives)
    
    @property
    def prefixes(self) -> set[str]:
        return {d.prefix for d in self.directives}


@dataclass
class ParsedFunction:
    """A fully parsed LLVM IR function with separated components"""
    name: str                              # @function_name
    signature: str                         # Full define line
    body_lines: list[str]                  # Pure code lines (no comments)
    check_blocks: list[CheckBlock]         # Groups of CHECK directives
    inline_comments: dict[int, str]        # line_idx -> comment (non-CHECK comments)
    raw_text: str                          # Original full text
    
    # Metadata for clustering
    has_vectors: bool = False
    has_scalable_vectors: bool = False     # vscale
    has_memory_ops: bool = False           # load/store
    has_control_flow: bool = False         # br, switch, phi
    has_calls: bool = False                # call, invoke
    has_atomics: bool = False              # atomic operations
    return_type: str = ""
    arg_types: list[str] = field(default_factory=list)
    
    # Source info
    source_file: Optional[str] = None
    
    def code_only(self) -> str:
        """Return just the code without any comments"""
        return self.signature + '\n' + '\n'.join(self.body_lines)
    
    def with_checks(self, new_checks: list[CheckBlock], new_name: Optional[str] = None) -> str:
        """Rebuild function with different CHECK blocks"""
        name = new_name or self.name
        lines = []
        
        # Add signature (with possibly new name)
        sig = self.signature
        if new_name and self.name in sig:
            sig = sig.replace(self.name, new_name)
        lines.append(sig)
        
        # Interleave code and checks
        # For simplicity, put all checks after signature, then code
        for block in new_checks:
            for directive in block.directives:
                # Update function name references in CHECK directives
                check_line = directive.raw_line
                if new_name and self.name in check_line:
                    check_line = check_line.replace(self.name, new_name)
                lines.append(check_line)
        
        # Add body
        for body_line in self.body_lines:
            if new_name and self.name in body_line:
                body_line = body_line.replace(self.name, new_name)
            lines.append(body_line)
            
        return '\n'.join(lines)


# Regex patterns
DEFINE_PATTERN = re.compile(
    r'define\s+'
    r'(?P<linkage>(?:private|internal|external|linkonce|weak|common|appending|'
    r'extern_weak|linkonce_odr|weak_odr|available_externally)?\s*)?'
    r'(?P<ret_type>[^@]+?)\s*'
    r'(?P<name>@[\w\d._-]+)\s*'
    r'\((?P<args>[^)]*)\)'
)

CHECK_PATTERN = re.compile(
    r';\s*(?P<prefix>[\w\d_-]+)-(?P<kind>LABEL|NEXT|SAME|NOT|DAG|COUNT-\d+|EMPTY|CHECK):\s*(?P<content>.*)'
)

# Simpler fallback for basic CHECK: directives
SIMPLE_CHECK_PATTERN = re.compile(
    r';\s*(?P<prefix>CHECK):\s*(?P<content>.*)'
)

VECTOR_TYPE_PATTERN = re.compile(r'<\d+\s*x\s*\w+>')
SCALABLE_VECTOR_PATTERN = re.compile(r'<vscale\s*x\s*\d+\s*x\s*\w+>')


def parse_check_line(line: str) -> Optional[CheckDirective]:
    """Parse a single line that might be a CHECK directive"""
    line_stripped = line.strip()
    if not line_stripped.startswith(';'):
        return None
    
    match = CHECK_PATTERN.match(line_stripped)
    if match:
        return CheckDirective(
            prefix=match.group('prefix'),
            kind=match.group('kind'),
            content=match.group('content'),
            raw_line=line
        )
    
    # Try simple CHECK: pattern
    match = SIMPLE_CHECK_PATTERN.match(line_stripped)
    if match:
        return CheckDirective(
            prefix='CHECK',
            kind='CHECK',
            content=match.group('content'),
            raw_line=line
        )
    
    return None


def extract_return_type(signature: str) -> str:
    """Extract return type from function signature"""
    match = DEFINE_PATTERN.match(signature.strip())
    if match:
        ret_type = match.group('ret_type').strip()
        # Clean up attributes that might be mixed in
        for attr in ['noundef', 'nonnull', 'signext', 'zeroext', 'inreg']:
            ret_type = ret_type.replace(attr, '').strip()
        return ret_type
    return "unknown"


def extract_arg_types(signature: str) -> list[str]:
    """Extract argument types from function signature"""
    match = DEFINE_PATTERN.match(signature.strip())
    if not match:
        return []
    
    args_str = match.group('args')
    if not args_str.strip():
        return []
    
    arg_types = []
    # Split by comma, but be careful with types like <4 x i32>
    depth = 0
    current = ""
    for char in args_str:
        if char == '<':
            depth += 1
        elif char == '>':
            depth -= 1
        elif char == ',' and depth == 0:
            arg_type = extract_type_from_arg(current.strip())
            if arg_type:
                arg_types.append(arg_type)
            current = ""
            continue
        current += char
    
    # Don't forget the last argument
    if current.strip():
        arg_type = extract_type_from_arg(current.strip())
        if arg_type:
            arg_types.append(arg_type)
    
    return arg_types


def extract_type_from_arg(arg: str) -> Optional[str]:
    """Extract just the type from an argument (type + name)"""
    # Handle things like: ptr noalias %data, <4 x i32> %vec, i32 noundef %x
    # The type is everything before the %name
    
    if '%' in arg:
        type_part = arg.split('%')[0].strip()
    else:
        type_part = arg.strip()
    
    # Remove attributes
    for attr in ['noalias', 'noundef', 'nonnull', 'signext', 'zeroext', 
                 'inreg', 'byval', 'sret', 'align', 'nocapture', 'readonly']:
        type_part = re.sub(rf'\b{attr}\b', '', type_part)
    
    # Remove alignment specs like align 4
    type_part = re.sub(r'align\s*\d+', '', type_part)
    
    return type_part.strip() or None


def extract_function_name(text: str) -> Optional[str]:
    """Extract @name from function definition"""
    match = re.search(r'define\s+[^@]*(@[\w\d._-]+)', text)
    if match:
        return match.group(1)
    return None


def analyze_operations(lines: list[str]) -> dict[str, bool]:
    """Analyze code lines to detect operation categories"""
    text = '\n'.join(lines)
    
    return {
        'has_vectors': bool(VECTOR_TYPE_PATTERN.search(text)),
        'has_scalable_vectors': bool(SCALABLE_VECTOR_PATTERN.search(text)),
        'has_memory_ops': bool(re.search(r'\b(load|store)\b', text)),
        'has_control_flow': bool(re.search(r'\b(br|switch|phi|select)\b', text)),
        'has_calls': bool(re.search(r'\b(call|invoke|tail call)\b', text)),
        'has_atomics': bool(re.search(r'\b(atomicrmw|cmpxchg|fence)\b', text)),
    }


def parse_function(func_text: str, source_file: Optional[str] = None) -> Optional[ParsedFunction]:
    """
    Parse a raw function string into a ParsedFunction.
    
    Separates code from CHECK directives, extracts metadata for clustering.
    """
    lines = func_text.split('\n')
    
    if not lines:
        return None
    
    # Find the signature (first line starting with 'define')
    signature = None
    sig_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('define '):
            signature = line
            sig_idx = i
            break
    
    if signature is None:
        return None
    
    name = extract_function_name(signature)
    if name is None:
        return None
    
    # Parse remaining lines
    body_lines = []
    check_blocks = []
    current_check_block = CheckBlock()
    inline_comments = {}
    
    for i, line in enumerate(lines[sig_idx + 1:], start=sig_idx + 1):
        # Try to parse as CHECK directive
        check = parse_check_line(line)
        
        if check:
            current_check_block.directives.append(check)
        else:
            # If we were accumulating checks, save the block
            if current_check_block.directives:
                check_blocks.append(current_check_block)
                current_check_block = CheckBlock()
            
            # Check if it's a regular comment
            stripped = line.strip()
            if stripped.startswith(';'):
                # Non-CHECK comment
                inline_comments[len(body_lines)] = line
            
            # Add to body (including empty lines and non-CHECK comments for structure)
            body_lines.append(line)
    
    # Don't forget the last check block
    if current_check_block.directives:
        check_blocks.append(current_check_block)
    
    # Analyze operations
    ops = analyze_operations(body_lines)
    
    # Extract types
    ret_type = extract_return_type(signature)
    arg_types = extract_arg_types(signature)
    
    return ParsedFunction(
        name=name,
        signature=signature,
        body_lines=body_lines,
        check_blocks=check_blocks,
        inline_comments=inline_comments,
        raw_text=func_text,
        has_vectors=ops['has_vectors'],
        has_scalable_vectors=ops['has_scalable_vectors'],
        has_memory_ops=ops['has_memory_ops'],
        has_control_flow=ops['has_control_flow'],
        has_calls=ops['has_calls'],
        has_atomics=ops['has_atomics'],
        return_type=ret_type,
        arg_types=arg_types,
        source_file=source_file,
    )


def parse_functions_from_file(file_path: Path) -> list[ParsedFunction]:
    """Extract and parse all functions from an LLVM IR file"""
    try:
        content = file_path.read_text()
    except OSError:
        return []
    
    functions = []
    lines = content.split('\n')
    func_lines = []
    in_func = False
    brace_count = 0
    
    for line in lines:
        if line.strip().startswith('define '):
            in_func = True
            func_lines = [line]
            brace_count = line.count('{') - line.count('}')
        elif in_func:
            func_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            
            if brace_count <= 0:
                func_text = '\n'.join(func_lines)
                parsed = parse_function(func_text, source_file=str(file_path))
                if parsed and 100 < len(func_text) < 2000:
                    functions.append(parsed)
                in_func = False
    
    return functions


def parse_functions_from_directory(test_dir: Path, max_count: int = 8000) -> list[ParsedFunction]:
    """Extract and parse functions from all .ll files in a directory"""
    functions = []
    
    for ll_file in test_dir.rglob("*.ll"):
        file_funcs = parse_functions_from_file(ll_file)
        functions.extend(file_funcs)
        
        if len(functions) >= max_count:
            return functions[:max_count]
    
    return functions


# Convenience function for quick testing
def parse_function_string(func_text: str) -> Optional[ParsedFunction]:
    """Parse a function from a raw string (for testing)"""
    return parse_function(func_text)