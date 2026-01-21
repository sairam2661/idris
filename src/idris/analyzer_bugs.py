#!/usr/bin/env python3
"""
Post-processing script for fuzzer output.

Analyzes bugs to find:
1. Bugs that fail on specific passes but NOT O2 (most interesting)
2. Deduplicate bugs that fail on many passes (likely FP or same root cause)
3. Rank bugs by "uniqueness" - fewer passes = more interesting

Usage:
    python analyze_bugs.py /path/to/output/bugs
    python analyze_bugs.py /path/to/output/bugs --detailed
    python analyze_bugs.py /path/to/output/bugs --move-fps /path/to/fps
"""

import argparse
import json
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class BugInfo:
    """Information about a single bug."""
    bug_dir: Path
    pass_name: str
    src_hash: str  # Hash of source IR for grouping
    strategy: str
    injected_type: str
    src_ir: str
    tgt_ir: str
    alive_output: str


@dataclass 
class BugGroup:
    """Group of bugs with the same source IR."""
    src_hash: str
    src_ir: str
    bugs: list[BugInfo] = field(default_factory=list)
    
    @property
    def passes(self) -> set[str]:
        return {b.pass_name for b in self.bugs}
    
    @property
    def has_o2(self) -> bool:
        return "default<O2>" in self.passes
    
    @property
    def num_passes(self) -> int:
        return len(self.passes)
    
    @property
    def specific_passes(self) -> set[str]:
        """Passes excluding O2."""
        return self.passes - {"default<O2>"}
    
    @property
    def strategy(self) -> str:
        return self.bugs[0].strategy if self.bugs else "unknown"
    
    @property
    def injected_type(self) -> str:
        return self.bugs[0].injected_type if self.bugs else "unknown"


def hash_ir(ir: str) -> str:
    """Hash IR for deduplication, ignoring function names."""
    # Normalize: remove function names which may differ
    import re
    normalized = re.sub(r'@[\w.]+', '@FUNC', ir)
    normalized = re.sub(r'%[\w.]+', '%VAR', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def load_bugs(bugs_dir: Path) -> list[BugInfo]:
    """Load all bugs from directory."""
    bugs = []
    
    for bug_dir in bugs_dir.iterdir():
        if not bug_dir.is_dir() or not bug_dir.name.startswith("bug_"):
            continue
        
        src_file = bug_dir / "src.ll"
        tgt_file = bug_dir / "tgt.ll"
        info_file = bug_dir / "info.json"
        alive_file = bug_dir / "alive_output.txt"
        
        if not src_file.exists():
            continue
        
        src_ir = src_file.read_text()
        tgt_ir = tgt_file.read_text() if tgt_file.exists() else ""
        alive_output = alive_file.read_text() if alive_file.exists() else ""
        
        # Parse info
        info = {}
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text())
            except:
                pass
        
        # Extract pass from directory name or info
        pass_name = info.get("pass", "unknown")
        if pass_name == "unknown":
            # Try to extract from dir name: bug_0001_instcombine
            parts = bug_dir.name.split("_")
            if len(parts) >= 3:
                pass_name = "_".join(parts[2:])
        
        bugs.append(BugInfo(
            bug_dir=bug_dir,
            pass_name=pass_name,
            src_hash=hash_ir(src_ir),
            strategy=info.get("strategy", "unknown"),
            injected_type=info.get("injected_type", "unknown"),
            src_ir=src_ir,
            tgt_ir=tgt_ir,
            alive_output=alive_output,
        ))
    
    return bugs


def group_bugs(bugs: list[BugInfo]) -> list[BugGroup]:
    """Group bugs by source IR hash."""
    groups = defaultdict(list)
    
    for bug in bugs:
        groups[bug.src_hash].append(bug)
    
    result = []
    for src_hash, bug_list in groups.items():
        group = BugGroup(
            src_hash=src_hash,
            src_ir=bug_list[0].src_ir,
            bugs=bug_list,
        )
        result.append(group)
    
    return result


def analyze_groups(groups: list[BugGroup]) -> dict:
    """Analyze bug groups and categorize them."""
    
    analysis = {
        "total_bugs": sum(len(g.bugs) for g in groups),
        "unique_sources": len(groups),
        "categories": {
            "pass_specific_no_o2": [],  # Most interesting: fails on pass but not O2
            "single_pass": [],           # Only one pass fails
            "few_passes": [],            # 2-3 passes fail
            "many_passes_no_o2": [],     # Many passes but not O2 - might be FP
            "all_including_o2": [],      # Fails on O2 too - likely real or known FP
        },
        "by_pass": defaultdict(int),
        "by_strategy": defaultdict(int),
        "by_type": defaultdict(int),
    }
    
    for group in groups:
        # Categorize
        if not group.has_o2:
            if group.num_passes == 1:
                analysis["categories"]["single_pass"].append(group)
            elif group.num_passes <= 3:
                analysis["categories"]["few_passes"].append(group)
                analysis["categories"]["pass_specific_no_o2"].append(group)
            else:
                analysis["categories"]["many_passes_no_o2"].append(group)
                analysis["categories"]["pass_specific_no_o2"].append(group)
        else:
            analysis["categories"]["all_including_o2"].append(group)
        
        # Count by pass
        for pass_name in group.passes:
            analysis["by_pass"][pass_name] += 1
        
        # Count by strategy/type
        analysis["by_strategy"][group.strategy] += 1
        analysis["by_type"][group.injected_type] += 1
    
    # Sort pass_specific by number of passes (fewer = more interesting)
    analysis["categories"]["pass_specific_no_o2"].sort(key=lambda g: g.num_passes)
    analysis["categories"]["single_pass"].sort(key=lambda g: list(g.passes)[0])
    
    return analysis


def print_analysis(analysis: dict, detailed: bool = False):
    """Print analysis results."""
    
    print(f"\n{'='*70}")
    print("BUG ANALYSIS")
    print(f"{'='*70}")
    print(f"Total bug reports: {analysis['total_bugs']}")
    print(f"Unique source IRs: {analysis['unique_sources']}")
    
    # Most interesting: pass-specific, not O2
    pass_specific = analysis["categories"]["pass_specific_no_o2"]
    single_pass = analysis["categories"]["single_pass"]
    few_passes = analysis["categories"]["few_passes"]
    many_no_o2 = analysis["categories"]["many_passes_no_o2"]
    with_o2 = analysis["categories"]["all_including_o2"]
    
    print(f"\n--- CATEGORIZATION ---")
    print(f"Pass-specific (no O2): {len(pass_specific)} unique bugs")
    print(f"  - Single pass only: {len(single_pass)}")
    print(f"  - 2-3 passes: {len(few_passes)}")
    print(f"  - Many passes (likely FP): {len(many_no_o2)}")
    print(f"Fails on O2 too: {len(with_o2)} unique bugs")
    
    # Single pass bugs - MOST INTERESTING
    if single_pass:
        print(f"\n--- SINGLE-PASS BUGS (Most Interesting!) ---")
        for group in single_pass[:20]:  # Top 20
            pass_name = list(group.passes)[0]
            print(f"  [{pass_name}] strategy={group.strategy} type={group.injected_type}")
            if detailed:
                print(f"    Source: {group.bugs[0].bug_dir}")
                print(f"    IR preview: {group.src_ir[:100].replace(chr(10), ' ')}...")
        if len(single_pass) > 20:
            print(f"  ... and {len(single_pass) - 20} more")
    
    # Few passes bugs
    if few_passes:
        print(f"\n--- FEW-PASS BUGS (2-3 passes, no O2) ---")
        for group in few_passes[:10]:
            passes = ", ".join(sorted(group.passes))
            print(f"  [{passes}] strategy={group.strategy} type={group.injected_type}")
            if detailed:
                print(f"    Source: {group.bugs[0].bug_dir}")
    
    # Likely FPs - many passes but no O2
    if many_no_o2:
        print(f"\n--- LIKELY FALSE POSITIVES (many passes, no O2) ---")
        print(f"  {len(many_no_o2)} unique sources fail on 4+ passes but not O2")
        if detailed:
            for group in many_no_o2[:5]:
                passes = ", ".join(sorted(group.passes))
                print(f"  [{group.num_passes} passes] {passes[:50]}...")
    
    # By pass breakdown
    print(f"\n--- BUGS BY PASS ---")
    sorted_passes = sorted(analysis["by_pass"].items(), key=lambda x: -x[1])
    for pass_name, count in sorted_passes[:15]:
        print(f"  {pass_name}: {count}")
    
    # By strategy
    print(f"\n--- BUGS BY STRATEGY ---")
    sorted_strategies = sorted(analysis["by_strategy"].items(), key=lambda x: -x[1])
    for strategy, count in sorted_strategies:
        print(f"  {strategy}: {count}")
    
    # By type
    print(f"\n--- BUGS BY TYPE ---")
    sorted_types = sorted(analysis["by_type"].items(), key=lambda x: -x[1])
    for typ, count in sorted_types[:15]:
        print(f"  {typ}: {count}")


def move_likely_fps(groups: list[BugGroup], analysis: dict, fp_dir: Path):
    """Move likely false positives to a separate directory."""
    
    fp_dir.mkdir(exist_ok=True)
    
    # Move bugs that fail on many passes but not O2
    many_no_o2 = analysis["categories"]["many_passes_no_o2"]
    
    moved = 0
    for group in many_no_o2:
        for bug in group.bugs:
            dest = fp_dir / bug.bug_dir.name
            if bug.bug_dir.exists():
                shutil.move(str(bug.bug_dir), str(dest))
                moved += 1
    
    print(f"\nMoved {moved} likely false positive bug reports to {fp_dir}")


def export_interesting(groups: list[BugGroup], analysis: dict, output_dir: Path):
    """Export most interesting bugs to a clean directory."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Export single-pass bugs
    single_pass_dir = output_dir / "single_pass"
    single_pass_dir.mkdir(exist_ok=True)
    
    for i, group in enumerate(analysis["categories"]["single_pass"]):
        pass_name = list(group.passes)[0]
        bug = group.bugs[0]
        
        dest_dir = single_pass_dir / f"{i:04d}_{pass_name}"
        dest_dir.mkdir(exist_ok=True)
        
        (dest_dir / "src.ll").write_text(bug.src_ir)
        (dest_dir / "tgt.ll").write_text(bug.tgt_ir)
        (dest_dir / "alive_output.txt").write_text(bug.alive_output)
        (dest_dir / "info.json").write_text(json.dumps({
            "pass": pass_name,
            "strategy": group.strategy,
            "injected_type": group.injected_type,
            "original_dir": str(bug.bug_dir),
        }, indent=2))
    
    print(f"\nExported {len(analysis['categories']['single_pass'])} single-pass bugs to {single_pass_dir}")
    
    # Export few-pass bugs
    few_pass_dir = output_dir / "few_passes"
    few_pass_dir.mkdir(exist_ok=True)
    
    for i, group in enumerate(analysis["categories"]["few_passes"]):
        passes = "_".join(sorted(group.passes))[:50]
        bug = group.bugs[0]
        
        dest_dir = few_pass_dir / f"{i:04d}_{passes}"
        dest_dir.mkdir(exist_ok=True)
        
        (dest_dir / "src.ll").write_text(bug.src_ir)
        (dest_dir / "tgt.ll").write_text(bug.tgt_ir)
        (dest_dir / "alive_output.txt").write_text(bug.alive_output)
        (dest_dir / "info.json").write_text(json.dumps({
            "passes": list(group.passes),
            "strategy": group.strategy,
            "injected_type": group.injected_type,
            "original_dir": str(bug.bug_dir),
        }, indent=2))
    
    print(f"Exported {len(analysis['categories']['few_passes'])} few-pass bugs to {few_pass_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and deduplicate fuzzer bugs")
    parser.add_argument("bugs_dir", type=Path, help="Path to bugs directory")
    parser.add_argument("--detailed", action="store_true", help="Show detailed output")
    parser.add_argument("--move-fps", type=Path, help="Move likely FPs to this directory")
    parser.add_argument("--export", type=Path, help="Export interesting bugs to this directory")
    
    args = parser.parse_args()
    
    if not args.bugs_dir.exists():
        print(f"Error: {args.bugs_dir} does not exist")
        return 1
    
    print(f"Loading bugs from {args.bugs_dir}...")
    bugs = load_bugs(args.bugs_dir)
    print(f"Loaded {len(bugs)} bug reports")
    
    if not bugs:
        print("No bugs found!")
        return 0
    
    print("Grouping by source IR...")
    groups = group_bugs(bugs)
    print(f"Found {len(groups)} unique source IRs")
    
    print("Analyzing...")
    analysis = analyze_groups(groups)
    
    print_analysis(analysis, detailed=args.detailed)
    
    if args.move_fps:
        move_likely_fps(groups, analysis, args.move_fps)
    
    if args.export:
        export_interesting(groups, analysis, args.export)
    
    return 0


if __name__ == "__main__":
    exit(main())