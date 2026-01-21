"""
Validation pipeline for generated LLVM IR.

Uses opt for verification and optimization, alive-tv for correctness checking.
Supports testing against multiple passes.
"""

import os
import tempfile
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from idris.config import get_config


# Comprehensive list of passes to test
ALL_PASSES = [
    # Core optimization passes (high bug potential)
    "instcombine",
    "gvn",
    "sccp",
    "dce",
    "adce",
    
    # Vectorization (historically buggy with unusual types)
    "slp-vectorizer",
    "loop-vectorize",
    
    # Loop transforms
    "licm",
    "loop-unroll",
    "loop-idiom",
    "loop-deletion",
    "indvars",
    
    # Memory optimizations
    "dse",
    "memcpyopt",
    "sroa",
    
    # Control flow
    "simplifycfg",
    "jump-threading",
    
    # Arithmetic
    "reassociate",
    "correlated-propagation",
    
    # Other useful passes
    "early-cse",
    "bdce",
    "aggressive-instcombine",
]

# Quick subset for faster iteration
QUICK_PASSES = [
    "instcombine",
    "gvn", 
    "slp-vectorizer",
    "loop-vectorize",
    "simplifycfg",
    "sccp",
    "instsimplify",
    "vector-combine"
]


def is_crash_signal(returncode: int) -> bool:
    """
    Check if return code indicates a crash (signal-based termination).
    """
    if returncode >= 0:
        return False
    crash_signals = {-6, -11, -8, -4, -7}
    return returncode in crash_signals or returncode < -1


def get_crash_type(returncode: int) -> str:
    """Get human-readable crash type from return code"""
    signal_names = {
        -6: "SIGABRT",
        -11: "SIGSEGV",
        -8: "SIGFPE",
        -4: "SIGILL",
        -7: "SIGBUS",
        -9: "SIGKILL",
        -15: "SIGTERM",
    }
    return signal_names.get(returncode, f"signal_{-returncode}")


def run_pass(ir: str, passes: str, opt_path: Path, timeout: int = 30) -> dict:
    """
    Run opt with given passes on IR.
    
    Args:
        ir: LLVM IR source
        passes: Pass pipeline string (e.g., "instcombine" or "default<O2>")
        opt_path: Path to opt binary
        timeout: Timeout in seconds
    
    Returns dict with:
        - success: bool
        - output: str | None - Optimized IR if successful
        - crash: bool
        - crash_type: str | None
        - timeout: bool
        - stderr: str
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
        f.write(ir)
        path = f.name
    
    try:
        result = subprocess.run(
            [str(opt_path), f'-passes={passes}', '-S', path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
                "crash": False,
                "crash_type": None,
                "timeout": False,
                "stderr": result.stderr,
            }
        elif is_crash_signal(result.returncode):
            return {
                "success": False,
                "output": None,
                "crash": True,
                "crash_type": get_crash_type(result.returncode),
                "timeout": False,
                "stderr": result.stderr,
            }
        else:
            return {
                "success": False,
                "output": None,
                "crash": False,
                "crash_type": None,
                "timeout": False,
                "stderr": result.stderr,
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": None,
            "crash": False,
            "crash_type": None,
            "timeout": True,
            "stderr": "timeout",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "output": None,
            "crash": False,
            "crash_type": None,
            "timeout": False,
            "stderr": f"opt not found at {opt_path}",
        }
    finally:
        if os.path.exists(path):
            os.unlink(path)


def verify_ir(ir: str, opt_path: Path) -> dict:
    """Verify that IR is syntactically valid using opt -passes=verify."""
    return run_pass(ir, "verify", opt_path, timeout=5)


def is_known_false_positive(alive_output: str, src_ir: str, tgt_ir: str) -> Optional[str]:
    """
    Check if an alive-tv "incorrect" result is a known false positive.
    
    Returns the reason string if it's a known FP, None otherwise.
    """
    output_lower = alive_output.lower()
    
    # Recursion - alive2 can't handle
    if "did not return" in output_lower:
        return "recursion"
    
    # Timeout during verification
    if "timeout" in output_lower:
        return "timeout"
    
    # initializes attribute not handled correctly
    if "initializes(" in tgt_ir:
        return "initializes_attr"
    
    # Add more patterns as discovered
    
    return None


def check_alive(src: str, tgt: str, alive_tv_path: Path, timeout: int = 60) -> dict:
    """
    Check if optimization is correct using alive-tv.
    
    Returns dict with keys: correct, incorrect, false_positive, 
    false_positive_reason, timeout, output
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f1:
        f1.write(src)
        src_path = f1.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f2:
        f2.write(tgt)
        tgt_path = f2.name
    
    try:
        result = subprocess.run(
            [str(alive_tv_path), src_path, tgt_path, "--disable-undef-input"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        
        is_incorrect = "Transformation doesn't verify" in output
        false_positive_reason = None
        
        if is_incorrect:
            false_positive_reason = is_known_false_positive(output, src, tgt)
        
        return {
            "correct": "Transformation seems to be correct" in output,
            "incorrect": is_incorrect and false_positive_reason is None,
            "false_positive": is_incorrect and false_positive_reason is not None,
            "false_positive_reason": false_positive_reason,
            "timeout": False,
            "output": output
        }
    except subprocess.TimeoutExpired:
        return {
            "correct": False,
            "incorrect": False,
            "false_positive": False,
            "false_positive_reason": None,
            "timeout": True,
            "output": "timeout"
        }
    except Exception as e:
        return {
            "correct": False,
            "incorrect": False,
            "false_positive": False,
            "false_positive_reason": None,
            "timeout": False,
            "output": str(e)
        }
    finally:
        os.unlink(src_path)
        os.unlink(tgt_path)


def should_reject_ir(ir: str) -> tuple[bool, str]:
    """
    Quick rejection of obviously problematic IR.
    
    Returns (should_reject, reason).
    """
    # Reject undef (but not poison - poison is useful for testing)
    if ' undef' in ir or ',undef' in ir or '(undef)' in ir:
        return True, "has_undef"
    
    return False, ""


def test_single_pass(ir: str, pass_name: str, opt_path: Path, 
                     alive_tv_path: Path) -> dict:
    """
    Test IR against a single optimization pass.
    
    Returns dict with outcome and details.
    """
    result = {
        "pass": pass_name,
        "outcome": None,
        "crash_type": None,
        "optimized_ir": None,
        "alive_output": None,
        "stderr": None,
    }
    
    # Run the pass
    opt_result = run_pass(ir, pass_name, opt_path)
    
    if opt_result["crash"]:
        result["outcome"] = "crash"
        result["crash_type"] = opt_result["crash_type"]
        result["stderr"] = opt_result["stderr"]
        return result
    
    if opt_result["timeout"]:
        result["outcome"] = "timeout_opt"
        return result
    
    if not opt_result["success"]:
        result["outcome"] = "opt_failed"
        result["stderr"] = opt_result["stderr"]
        return result
    
    optimized = opt_result["output"]
    result["optimized_ir"] = optimized
    
    # Check with alive2
    alive = check_alive(ir, optimized, alive_tv_path)
    
    if alive["correct"]:
        result["outcome"] = "correct"
        return result
    
    if alive["timeout"]:
        result["outcome"] = "timeout_alive"
        return result
    
    if alive["false_positive"]:
        result["outcome"] = f"false_positive_{alive['false_positive_reason']}"
        result["alive_output"] = alive["output"]
        return result
    
    if alive["incorrect"]:
        result["outcome"] = "miscompile"
        result["alive_output"] = alive["output"]
        return result
    
    result["outcome"] = "unknown"
    return result


def test_multi_pass(ir: str, passes: list[str], opt_path: Path,
                    alive_tv_path: Path, stop_on_bug: bool = False) -> list[dict]:
    """
    Test IR against multiple passes.
    
    Args:
        ir: Source IR
        passes: List of pass names to test
        opt_path: Path to opt
        alive_tv_path: Path to alive-tv
        stop_on_bug: If True, stop testing after first crash/miscompile
    
    Returns list of results, one per pass tested.
    """
    results = []
    
    for pass_name in passes:
        result = test_single_pass(ir, pass_name, opt_path, alive_tv_path)
        results.append(result)
        
        # Early exit on crash or miscompile if requested
        if stop_on_bug and result["outcome"] in ("crash", "miscompile"):
            break
    
    return results


def multi_pass_validation_worker(
    item: tuple,
    passes: list[str],
    output_dirs: dict,
    stats: dict,
    stats_lock,
) -> list[dict]:
    """
    Worker function for validating IR against multiple passes.
    
    Args:
        item: Tuple of (ir, metadata)
        passes: List of passes to test
        output_dirs: Dict with paths for bugs, crashes, valid, fp
        stats: Shared statistics dict
        stats_lock: Lock for stats
    
    Returns list of test results.
    """
    cfg = get_config()
    opt_path = Path(cfg["paths"]["opt"])
    alive_tv_path = Path(cfg["paths"]["alive_tv"])
    
    ir, metadata = item
    strategy = metadata.get("strategy", "unknown")
    injected_type = metadata.get("injected_type", "unknown")
    
    # Quick reject
    reject, reason = should_reject_ir(ir)
    if reject:
        with stats_lock:
            stats["rejected"] = stats.get("rejected", 0) + 1
            stats[f"rejected_{reason}"] = stats.get(f"rejected_{reason}", 0) + 1
        return []
    
    # Verify IR is valid first
    verify_result = verify_ir(ir, opt_path)
    
    if verify_result["crash"]:
        with stats_lock:
            stats["crashes"] = stats.get("crashes", 0) + 1
            crash_id = stats["crashes"]
            stats["crashes_verify"] = stats.get("crashes_verify", 0) + 1
        
        # Save crash
        crash_dir = output_dirs["crashes"] / f"crash_{crash_id:04d}_verify"
        crash_dir.mkdir(exist_ok=True)
        (crash_dir / "input.ll").write_text(ir)
        (crash_dir / "info.json").write_text(json.dumps({
            "pass": "verify",
            "crash_type": verify_result["crash_type"],
            "strategy": strategy,
            "injected_type": injected_type,
            "timestamp": datetime.now().isoformat(),
        }, indent=2))
        if verify_result.get("stderr"):
            (crash_dir / "stderr.txt").write_text(verify_result["stderr"])
        
        print(f"\n[CRASH #{crash_id}] verify: {verify_result['crash_type']} [{strategy}]")
        return [{"pass": "verify", "outcome": "crash", "crash_type": verify_result["crash_type"]}]
    
    if not verify_result["success"]:
        with stats_lock:
            stats["invalid"] = stats.get("invalid", 0) + 1
        return [{"pass": "verify", "outcome": "invalid"}]
    
    # Track valid IR
    with stats_lock:
        stats["valid"] = stats.get("valid", 0) + 1
        valid_id = stats["valid"]
    
    (output_dirs["valid"] / f"valid_{valid_id:06d}.ll").write_text(ir)
    
    # Test against all passes
    results = []
    
    for pass_name in passes:
        result = test_single_pass(ir, pass_name, opt_path, alive_tv_path)
        result["metadata"] = metadata
        results.append(result)
        
        outcome = result["outcome"]
        
        if outcome == "crash":
            with stats_lock:
                stats["crashes"] = stats.get("crashes", 0) + 1
                crash_id = stats["crashes"]
                
                # Track by pass
                key = f"crashes_{pass_name}"
                stats[key] = stats.get(key, 0) + 1
                
                # Track by strategy
                key = f"crashes_strategy_{strategy}"
                stats[key] = stats.get(key, 0) + 1
            
            # Save crash
            crash_dir = output_dirs["crashes"] / f"crash_{crash_id:04d}_{pass_name}"
            crash_dir.mkdir(exist_ok=True)
            (crash_dir / "input.ll").write_text(ir)
            (crash_dir / "info.json").write_text(json.dumps({
                "pass": pass_name,
                "crash_type": result["crash_type"],
                "strategy": strategy,
                "injected_type": injected_type,
                "timestamp": datetime.now().isoformat(),
            }, indent=2))
            if result.get("stderr"):
                (crash_dir / "stderr.txt").write_text(result["stderr"])
            
            print(f"\n[CRASH #{crash_id}] {pass_name}: {result['crash_type']} [{strategy}] type={injected_type}")
        
        elif outcome == "miscompile":
            with stats_lock:
                stats["miscompiles"] = stats.get("miscompiles", 0) + 1
                bug_id = stats["miscompiles"]
                
                # Track by pass
                key = f"miscompiles_{pass_name}"
                stats[key] = stats.get(key, 0) + 1
                
                # Track by strategy
                key = f"miscompiles_strategy_{strategy}"
                stats[key] = stats.get(key, 0) + 1
            
            # Save bug
            bug_dir = output_dirs["bugs"] / f"bug_{bug_id:04d}_{pass_name}"
            bug_dir.mkdir(exist_ok=True)
            (bug_dir / "src.ll").write_text(ir)
            (bug_dir / "tgt.ll").write_text(result.get("optimized_ir", ""))
            (bug_dir / "alive_output.txt").write_text(result.get("alive_output", ""))
            (bug_dir / "info.json").write_text(json.dumps({
                "pass": pass_name,
                "strategy": strategy,
                "injected_type": injected_type,
                "timestamp": datetime.now().isoformat(),
            }, indent=2))
            
            print(f"\n[MISCOMPILE #{bug_id}] {pass_name} [{strategy}] type={injected_type}")
        
        elif outcome == "correct":
            with stats_lock:
                stats["correct"] = stats.get("correct", 0) + 1
        
        elif outcome and outcome.startswith("false_positive"):
            with stats_lock:
                stats["false_positives"] = stats.get("false_positives", 0) + 1
                fp_id = stats["false_positives"]
            
            # Save FP for analysis
            reason = outcome.replace("false_positive_", "")
            fp_subdir = output_dirs["fp"] / reason
            fp_subdir.mkdir(exist_ok=True)
            
            fp_dir = fp_subdir / f"fp_{fp_id:04d}_{pass_name}"
            fp_dir.mkdir(exist_ok=True)
            (fp_dir / "src.ll").write_text(ir)
            (fp_dir / "tgt.ll").write_text(result.get("optimized_ir", ""))
            (fp_dir / "alive_output.txt").write_text(result.get("alive_output", ""))
        
        elif outcome and "timeout" in outcome:
            with stats_lock:
                stats["timeouts"] = stats.get("timeouts", 0) + 1
    
    return results


# Keep old interface for backwards compatibility
def validation_worker(item, bugs_dir, valid_dir, fp_dir, stats, stats_lock, 
                      perplexity_log, ppl_lock):
    """
    Legacy worker function - wraps multi_pass_validation_worker for O2 only.
    """
    ir, prompt, prompt_type, ppl_info = item
    
    # Convert to new format
    metadata = {
        "strategy": ppl_info.get("strategy", prompt_type),
        "injected_type": ppl_info.get("injected_type"),
        "prompt": prompt,
    }
    metadata.update(ppl_info)
    
    output_dirs = {
        "bugs": bugs_dir,
        "crashes": bugs_dir.parent / "crashes",
        "valid": valid_dir,
        "fp": fp_dir,
    }
    output_dirs["crashes"].mkdir(exist_ok=True)
    
    # Test O2 only for backwards compatibility
    results = multi_pass_validation_worker(
        (ir, metadata),
        passes=["default<O2>"],
        output_dirs=output_dirs,
        stats=stats,
        stats_lock=stats_lock,
    )
    
    # Log to perplexity log if provided
    if ppl_lock and perplexity_log is not None:
        record = {
            "prompt_type": prompt_type,
            "ir_length": len(ir),
            "perplexity": ppl_info.get("perplexity"),
            "mean_logprob": ppl_info.get("mean_logprob"),
            "strategy": metadata["strategy"],
            "outcome": results[0]["outcome"] if results else "rejected",
        }
        with ppl_lock:
            perplexity_log.append(record)
    
    return results[0]["outcome"] if results else "rejected"