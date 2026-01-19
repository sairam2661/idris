"""
Validation pipeline for generated LLVM IR.

Uses opt for verification and optimization, alive-tv for correctness checking.
"""

import os
import tempfile
import subprocess
import json
from datetime import datetime
from pathlib import Path

from idris.config import get_config


def is_crash_signal(returncode: int) -> bool:
    """
    Check if return code indicates a crash (signal-based termination).
    
    On Unix, negative return codes indicate the process was killed by a signal.
    Common crash signals:
      -6  (SIGABRT) - abort/assertion failure
      -11 (SIGSEGV) - segmentation fault
      -8  (SIGFPE)  - floating point exception
      -4  (SIGILL)  - illegal instruction
      -7  (SIGBUS)  - bus error
    """
    if returncode >= 0:
        return False
    
    # Negative return code = killed by signal
    crash_signals = {-6, -11, -8, -4, -7}  # ABRT, SEGV, FPE, ILL, BUS
    return returncode in crash_signals or returncode < -1


def get_crash_type(returncode: int) -> str:
    """Get human-readable crash type from return code"""
    signal_names = {
        -6: "SIGABRT (assertion/abort)",
        -11: "SIGSEGV (segfault)",
        -8: "SIGFPE (floating point)",
        -4: "SIGILL (illegal instruction)",
        -7: "SIGBUS (bus error)",
        -9: "SIGKILL (killed)",
        -15: "SIGTERM (terminated)",
    }
    return signal_names.get(returncode, f"signal {-returncode}")


def verify_ir(ir: str, opt_path: Path) -> dict:
    """
    Verify that IR is syntactically valid using opt.
    
    Returns dict with:
        - valid: bool - True if IR is valid
        - crash: bool - True if opt crashed
        - crash_type: str | None - Type of crash if crashed
        - timeout: bool - True if verification timed out
        - stderr: str - Error output from opt
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
        f.write(ir)
        path = f.name
    
    try:
        result = subprocess.run(
            [str(opt_path), '-passes=verify', '-S', path],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return {
                "valid": True,
                "crash": False,
                "crash_type": None,
                "timeout": False,
                "stderr": result.stderr,
            }
        elif is_crash_signal(result.returncode):
            return {
                "valid": False,
                "crash": True,
                "crash_type": get_crash_type(result.returncode),
                "timeout": False,
                "stderr": result.stderr,
            }
        else:
            # Normal failure (invalid IR)
            return {
                "valid": False,
                "crash": False,
                "crash_type": None,
                "timeout": False,
                "stderr": result.stderr,
            }
    except subprocess.TimeoutExpired:
        return {
            "valid": False,
            "crash": False,
            "crash_type": None,
            "timeout": True,
            "stderr": "timeout during verification",
        }
    except FileNotFoundError:
        print(f"Error: {opt_path} not found.")
        return {
            "valid": False,
            "crash": False,
            "crash_type": None,
            "timeout": False,
            "stderr": f"opt not found at {opt_path}",
        }
    finally:
        if os.path.exists(path):
            os.unlink(path)


def optimize_ir(ir: str, opt_path: Path) -> dict:
    """
    Run optimization passes on IR.
    
    Returns dict with:
        - success: bool - True if optimization succeeded
        - output: str | None - Optimized IR if successful
        - crash: bool - True if opt crashed
        - crash_type: str | None - Type of crash if crashed
        - timeout: bool - True if optimization timed out
        - stderr: str - Error output from opt
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
        f.write(ir)
        path = f.name
    
    try:
        result = subprocess.run(
            [str(opt_path), '-passes=default<O2>', '-S', path],
            capture_output=True,
            text=True,
            timeout=30
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
            "stderr": "timeout during optimization",
        }
    except FileNotFoundError:
        print(f"Error: {opt_path} not found.")
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


def is_known_false_positive(alive_output: str, tgt_ir: str) -> str | None:
    """
    Check if an alive-tv "incorrect" result is a known false positive.
    
    Returns the reason string if it's a known FP, None otherwise.
    """
    # Issue #1202: initializes attribute not handled correctly
    if "initializes(" in tgt_ir:
        return "initializes_attr"
    
    # Add more patterns as they're discovered:
    # if "some_pattern" in alive_output or "some_pattern" in tgt_ir:
    #     return "pattern_name"
    
    return None


def check_alive(src: str, tgt: str, alive_tv_path: Path) -> dict:
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
            timeout=40
        )
        output = result.stdout + result.stderr
        
        is_incorrect = "Transformation doesn't verify" in output
        false_positive_reason = None
        
        if is_incorrect:
            false_positive_reason = is_known_false_positive(output, tgt)
        
        return {
            "correct": "Transformation seems to be correct" in output,
            "incorrect": is_incorrect and false_positive_reason is None,
            "false_positive": is_incorrect and false_positive_reason is not None,
            "false_positive_reason": false_positive_reason,
            "timeout": "timeout" in output.lower(),
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

def validation_worker(item, bugs_dir, valid_dir, fp_dir, stats, stats_lock, 
                      perplexity_log, ppl_lock):
    """
    Worker function for validating a single generated IR.
    
    Args:
        item: Tuple of (ir, prompt, prompt_type, ppl_info)
        bugs_dir: Directory to save bug cases
        valid_dir: Directory to save valid IR
        fp_dir: Directory to save false positives
        stats: Shared statistics dict
        stats_lock: Lock for stats
        perplexity_log: Shared perplexity log list
        ppl_lock: Lock for perplexity log
    
    Returns:
        Outcome string: "invalid", "opt_failed", "correct", "bug", 
                       "false_positive", "timeout", "crash_verify", 
                       "crash_opt", "unknown"
    """
    cfg = get_config()
    opt_path = Path(cfg["paths"]["opt"])
    alive_tv_path = Path(cfg["paths"]["alive_tv"])
    
    ir, prompt, prompt_type, ppl_info = item
    
    # Extract weird seed info
    is_weird_seed = ppl_info.get("is_weird_seed", False)
    seed_perplexity = ppl_info.get("seed_perplexity")
    
    # Build record for perplexity log
    record = {
        "prompt_type": prompt_type,
        "ir_length": len(ir),
        "perplexity": ppl_info.get("perplexity"),
        "mean_logprob": ppl_info.get("mean_logprob"),
        "min_logprob": ppl_info.get("min_logprob"),
        "num_tokens": ppl_info.get("num_tokens"),
        "mutation_strategy": ppl_info.get("mutation_strategy"),
        "seed_has_vectors": ppl_info.get("seed_has_vectors"),
        "is_weird_seed": is_weird_seed,
        "seed_perplexity": seed_perplexity,
        "outcome": None,
    }
    
    # Step 1: Verify IR is valid
    verify_result = verify_ir(ir, opt_path)
    
    # Check for crash during verification
    if verify_result["crash"]:
        with stats_lock:
            if "crashes_verify" not in stats:
                stats["crashes_verify"] = 0
            stats["crashes_verify"] += 1
            crash_id = stats["crashes_verify"]
        
        # Save crash case
        crashes_dir = bugs_dir.parent / "crashes" / "verify"
        crashes_dir.mkdir(parents=True, exist_ok=True)
        
        crash_dir = crashes_dir / f"crash_{crash_id:04d}"
        crash_dir.mkdir(exist_ok=True)
        (crash_dir / "input.ll").write_text(ir)
        (crash_dir / "prompt.txt").write_text(prompt)
        (crash_dir / "stderr.txt").write_text(verify_result["stderr"])
        (crash_dir / "info.json").write_text(json.dumps({
            "crash_type": verify_result["crash_type"],
            "prompt_type": prompt_type,
            "timestamp": datetime.now().isoformat(),
            "stage": "verify",
            "perplexity": ppl_info.get("perplexity"),
            "mutation_strategy": ppl_info.get("mutation_strategy"),
            "is_weird_seed": is_weird_seed,
            "seed_perplexity": seed_perplexity,
        }))
        
        print(f"\n[CRASH {crash_id}] opt crashed during verify! {verify_result['crash_type']}")
        print(f"Input:\n{ir[:200]}...")
        
        record["outcome"] = f"crash_verify_{verify_result['crash_type']}"
        with ppl_lock:
            perplexity_log.append(record)
        return "crash_verify"
    
    if verify_result["timeout"]:
        record["outcome"] = "timeout_verify"
        with ppl_lock:
            perplexity_log.append(record)
        return "timeout_verify"
    
    if not verify_result["valid"]:
        record["outcome"] = "invalid"
        with ppl_lock:
            perplexity_log.append(record)
        return "invalid"
    
    # Track valid IR
    valid_id = None
    with stats_lock:
        stats["valid"] += 1
        valid_id = stats["valid"]
    
    (valid_dir / f"valid_{valid_id:06d}.ll").write_text(ir)
    
    # Step 2: Optimize
    opt_result = optimize_ir(ir, opt_path)
    
    # Check for crash during optimization
    if opt_result["crash"]:
        with stats_lock:
            if "crashes_opt" not in stats:
                stats["crashes_opt"] = 0
            stats["crashes_opt"] += 1
            crash_id = stats["crashes_opt"]
        
        # Save crash case
        crashes_dir = bugs_dir.parent / "crashes" / "optimize"
        crashes_dir.mkdir(parents=True, exist_ok=True)
        
        crash_dir = crashes_dir / f"crash_{crash_id:04d}"
        crash_dir.mkdir(exist_ok=True)
        (crash_dir / "input.ll").write_text(ir)
        (crash_dir / "prompt.txt").write_text(prompt)
        (crash_dir / "stderr.txt").write_text(opt_result["stderr"])
        (crash_dir / "info.json").write_text(json.dumps({
            "crash_type": opt_result["crash_type"],
            "prompt_type": prompt_type,
            "timestamp": datetime.now().isoformat(),
            "stage": "optimize",
            "passes": "default<O2>",
            "perplexity": ppl_info.get("perplexity"),
            "mutation_strategy": ppl_info.get("mutation_strategy"),
            "is_weird_seed": is_weird_seed,
            "seed_perplexity": seed_perplexity,
        }))
        
        print(f"\n[CRASH {crash_id}] opt crashed during optimization! {opt_result['crash_type']}")
        print(f"Input:\n{ir[:200]}...")
        
        record["outcome"] = f"crash_opt_{opt_result['crash_type']}"
        with ppl_lock:
            perplexity_log.append(record)
        return "crash_opt"
    
    if opt_result["timeout"]:
        with stats_lock:
            stats["timeout"] += 1
        record["outcome"] = "timeout_opt"
        with ppl_lock:
            perplexity_log.append(record)
        return "timeout_opt"
    
    if not opt_result["success"]:
        record["outcome"] = "opt_failed"
        with ppl_lock:
            perplexity_log.append(record)
        return "opt_failed"
    
    optimized = opt_result["output"]
    
    with stats_lock:
        stats["optimized"] += 1
    
    # Step 3: Check correctness with alive-tv
    result = check_alive(ir, optimized, alive_tv_path)
    
    if result["correct"]:
        with stats_lock:
            stats["correct"] += 1
        record["outcome"] = "correct"
        with ppl_lock:
            perplexity_log.append(record)
        return "correct"
    
    elif result["false_positive"]:
        with stats_lock:
            stats["false_positives"] += 1
            fp_id = stats["false_positives"]
        
        reason = result["false_positive_reason"]
        fp_subdir = fp_dir / reason
        fp_subdir.mkdir(exist_ok=True)
        
        case_dir = fp_subdir / f"fp_{fp_id:04d}"
        case_dir.mkdir(exist_ok=True)
        (case_dir / "src.ll").write_text(ir)
        (case_dir / "tgt.ll").write_text(optimized)
        (case_dir / "alive_output.txt").write_text(result["output"])
        
        record["outcome"] = f"false_positive_{reason}"
        with ppl_lock:
            perplexity_log.append(record)
        return "false_positive"
    
    elif result["incorrect"]:
        with stats_lock:
            stats["bugs"] += 1
            bug_id = stats["bugs"]
            
            # Track by seed weirdness
            if is_weird_seed:
                stats["bugs_from_weird_seeds"] = stats.get("bugs_from_weird_seeds", 0) + 1
            else:
                stats["bugs_from_normal_seeds"] = stats.get("bugs_from_normal_seeds", 0) + 1
            
            # NEW: Track by strategy
            strategy = ppl_info.get("strategy", prompt_type)
            if "by_strategy" in stats and strategy in stats["by_strategy"]:
                stats["by_strategy"][strategy]["bugs"] += 1
            
            # NEW: Track scalable vectors
            if ppl_info.get("seed_has_scalable_vectors"):
                stats["bugs_from_scalable_vectors"] = stats.get("bugs_from_scalable_vectors", 0) + 1
        
        bug_dir = bugs_dir / f"bug_{bug_id:04d}"
        bug_dir.mkdir(exist_ok=True)
        (bug_dir / "src.ll").write_text(ir)
        (bug_dir / "tgt.ll").write_text(optimized)
        (bug_dir / "alive_output.txt").write_text(result["output"])
        (bug_dir / "prompt.txt").write_text(prompt)
        (bug_dir / "info.json").write_text(json.dumps({
            "prompt_type": prompt_type,
            "timestamp": datetime.now().isoformat(),
            "perplexity": ppl_info.get("perplexity"),
            "mean_logprob": ppl_info.get("mean_logprob"),
            "min_logprob": ppl_info.get("min_logprob"),
            "num_tokens": ppl_info.get("num_tokens"),
            "strategy": ppl_info.get("strategy"),  # NEW
            "pattern_name": ppl_info.get("pattern_name"),  # NEW
            "is_weird_seed": is_weird_seed,
            "seed_perplexity": seed_perplexity,
            "seed_has_scalable_vectors": ppl_info.get("seed_has_scalable_vectors"),  # NEW
        }))
        
        # Updated print with strategy info
        ppl_str = f" PPL:{ppl_info.get('perplexity'):.2f}" if ppl_info.get('perplexity') else ""
        strategy_str = f" [{ppl_info.get('strategy', 'unknown')}]"
        weird_str = " [WEIRD]" if is_weird_seed else ""
        scalable_str = " [SCALABLE]" if ppl_info.get("seed_has_scalable_vectors") else ""
        print(f"\n[BUG {bug_id}] Miscompilation!{ppl_str}{strategy_str}{weird_str}{scalable_str}")
        print(f"Source:\n{ir[:200]}...")
        
    elif result["timeout"]:
        with stats_lock:
            stats["timeout"] += 1
        record["outcome"] = "timeout_alive"
        with ppl_lock:
            perplexity_log.append(record)
        return "timeout_alive"
    
    record["outcome"] = "unknown"
    with ppl_lock:
        perplexity_log.append(record)
    return "unknown"