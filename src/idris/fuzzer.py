"""
LLVM IR Fuzzer - Diverse Generation with Multi-Pass Testing

Generates diverse test cases using LLM completion and tests them
against multiple LLVM optimization passes to find crashes and miscompilations.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import json

from vllm import LLM, SamplingParams

from idris.config import get_config
from idris.generator import DiverseGenerator, build_corpus, Strategy
from idris.validation import (
    multi_pass_validation_worker,
    ALL_PASSES,
    QUICK_PASSES,
)
from idris.utils import complete_function


def run_fuzzer():
    """Main fuzzing loop with diverse generation and multi-pass testing."""
    
    cfg = get_config()
    
    # Paths
    llvm_test_dir = Path(cfg["paths"]["llvm_test_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    
    # Fuzzer settings
    batch_size = cfg["fuzzer"]["batch_size"]
    num_iterations = cfg["fuzzer"]["num_iterations"]
    temperature = cfg["fuzzer"]["temperature"]
    max_tokens = cfg["fuzzer"]["max_tokens"]
    num_workers = cfg["fuzzer"]["num_workers"]
    model_name = cfg["fuzzer"]["model"]
    
    # Pass selection
    use_quick_passes = cfg["fuzzer"].get("quick_passes", False)
    passes = QUICK_PASSES if use_quick_passes else ALL_PASSES
    
    # Also always include O2 for comparison
    if "default<O2>" not in passes:
        passes = passes + ["default<O2>"]
    
    # Setup output directories
    output_dir.mkdir(exist_ok=True)
    output_dirs = {
        "bugs": output_dir / "bugs",
        "crashes": output_dir / "crashes",
        "valid": output_dir / "valid",
        "fp": output_dir / "false_positives",
    }
    for d in output_dirs.values():
        d.mkdir(exist_ok=True)
    
    # Build corpus from LLVM test directory
    print("Building corpus from test directory...")
    corpus = build_corpus(llvm_test_dir, max_files=10000)
    print(f"Built corpus with {len(corpus)} test files")
    
    # Create diverse generator
    generator = DiverseGenerator(corpus)
    gen_stats = generator.stats()
    print(f"\nGenerator statistics:")
    print(f"  Test files: {gen_stats['total_test_files']}")
    print(f"  Passes covered: {gen_stats['num_passes']}")
    print(f"  With vectors: {gen_stats['with_vectors']}")
    print(f"  With scalable vectors: {gen_stats['with_scalable']}")
    print(f"  With loops: {gen_stats['with_loops']}")
    
    (output_dir / "generator_stats.json").write_text(json.dumps(gen_stats, indent=2, default=str))
    
    # Load LLM
    print(f"\nLoading {model_name}...")
    llm = LLM(model=model_name, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
    )
    
    # Statistics tracking
    stats = {
        "generated": 0,
        "valid": 0,
        "invalid": 0,
        "rejected": 0,
        "crashes": 0,
        "miscompiles": 0,
        "correct": 0,
        "false_positives": 0,
        "timeouts": 0,
    }
    stats_lock = threading.Lock()
    
    # Thread pool for validation
    executor = ThreadPoolExecutor(max_workers=num_workers)
    pending_futures = []
    
    print(f"\nStarting fuzzing...")
    print(f"Testing {len(passes)} passes: {', '.join(passes[:5])}{'...' if len(passes) > 5 else ''}")
    print(f"Batch size: {batch_size}, Iterations: {num_iterations}")
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Generate batch of prompts
        batch = generator.generate_batch(batch_size)
        prompts = [p for p, _ in batch]
        metadata_list = [m for _, m in batch]
        
        # Generate completions with LLM
        outputs = llm.generate(prompts, sampling_params)
        
        # Process completions
        submitted = 0
        for idx, output in enumerate(outputs):
            prompt = prompts[idx]
            metadata = metadata_list[idx]
            completion = output.outputs[0].text
            
            # Skip very short completions
            if len(completion) < 20:
                continue
            
            # Try to complete the function (balance braces)
            ir = complete_function(prompt, completion)
            if ir is None:
                continue
            
            submitted += 1
            
            # Submit for multi-pass validation
            future = executor.submit(
                multi_pass_validation_worker,
                (ir, metadata),
                passes,
                output_dirs,
                stats,
                stats_lock,
            )
            pending_futures.append(future)
        
        with stats_lock:
            stats["generated"] += len(outputs)
        
        # Clean up completed futures
        done = [f for f in pending_futures if f.done()]
        pending_futures = [f for f in pending_futures if not f.done()]
        
        # Progress update
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        
        with stats_lock:
            print(f"[{iteration+1}/{num_iterations}] {iter_time:.1f}s | "
                  f"Gen:{stats['generated']} Valid:{stats['valid']} "
                  f"Crashes:{stats['crashes']} Bugs:{stats['miscompiles']} "
                  f"FP:{stats['false_positives']} Pending:{len(pending_futures)}")
        
        # Periodic save
        if iteration % 10 == 0:
            save_stats(stats, stats_lock, output_dir, elapsed, iteration, passes)
    
    # Wait for remaining validations
    print("\nWaiting for pending validations...")
    for f in as_completed(pending_futures):
        pass
    
    executor.shutdown()
    
    # Final save
    elapsed = time.time() - start_time
    save_stats(stats, stats_lock, output_dir, elapsed, num_iterations, passes, completed=True)
    
    # Print summary
    print_summary(stats, elapsed, passes)


def save_stats(stats, stats_lock, output_dir, elapsed, iteration, passes, completed=False):
    """Save current statistics to file."""
    with stats_lock:
        stats_copy = dict(stats)
    
    stats_copy["elapsed_seconds"] = elapsed
    stats_copy["iteration"] = iteration
    stats_copy["passes_tested"] = passes
    stats_copy["completed"] = completed
    
    (output_dir / "stats.json").write_text(json.dumps(stats_copy, indent=2, default=str))


def print_summary(stats, elapsed, passes):
    """Print final summary."""
    print(f"\n{'='*70}")
    print("FUZZING COMPLETE")
    print(f"{'='*70}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total generated: {stats['generated']}")
    print(f"Valid IR: {stats['valid']}")
    print(f"Invalid/Rejected: {stats.get('invalid', 0) + stats.get('rejected', 0)}")
    print(f"\nBugs found:")
    print(f"  Crashes: {stats['crashes']}")
    print(f"  Miscompiles: {stats['miscompiles']}")
    print(f"  False positives: {stats['false_positives']}")
    print(f"  Timeouts: {stats['timeouts']}")
    
    # Print per-pass breakdown
    print(f"\nCrashes by pass:")
    for pass_name in passes:
        key = f"crashes_{pass_name}"
        count = stats.get(key, 0)
        if count > 0:
            print(f"  {pass_name}: {count}")
    
    print(f"\nMiscompiles by pass:")
    for pass_name in passes:
        key = f"miscompiles_{pass_name}"
        count = stats.get(key, 0)
        if count > 0:
            print(f"  {pass_name}: {count}")
    
    # Print by strategy
    print(f"\nCrashes by strategy:")
    for key, value in stats.items():
        if key.startswith("crashes_strategy_") and value > 0:
            strategy = key.replace("crashes_strategy_", "")
            print(f"  {strategy}: {value}")
    
    print(f"\nMiscompiles by strategy:")
    for key, value in stats.items():
        if key.startswith("miscompiles_strategy_") and value > 0:
            strategy = key.replace("miscompiles_strategy_", "")
            print(f"  {strategy}: {value}")


def run_on_existing(valid_dir: Path):
    """
    Run multi-pass testing on existing valid IR files.
    
    Useful for finding bugs in IR that was previously only tested with O2.
    """
    cfg = get_config()
    output_dir = Path(cfg["paths"]["output_dir"]) / "retest"
    
    output_dir.mkdir(exist_ok=True)
    output_dirs = {
        "bugs": output_dir / "bugs",
        "crashes": output_dir / "crashes", 
        "valid": output_dir / "valid",
        "fp": output_dir / "false_positives",
    }
    for d in output_dirs.values():
        d.mkdir(exist_ok=True)
    
    passes = QUICK_PASSES
    
    stats = {
        "tested": 0,
        "crashes": 0,
        "miscompiles": 0,
        "correct": 0,
        "false_positives": 0,
        "timeouts": 0,
    }
    stats_lock = threading.Lock()
    
    ll_files = list(valid_dir.glob("*.ll"))
    print(f"Found {len(ll_files)} IR files to test")
    print(f"Testing {len(passes)} passes...")
    
    executor = ThreadPoolExecutor(max_workers=cfg["fuzzer"]["num_workers"])
    futures = []
    
    for ll_file in ll_files:
        ir = ll_file.read_text()
        metadata = {
            "strategy": "existing",
            "source_file": str(ll_file),
        }
        
        future = executor.submit(
            multi_pass_validation_worker,
            (ir, metadata),
            passes,
            output_dirs,
            stats,
            stats_lock,
        )
        futures.append(future)
    
    # Wait with progress
    for i, f in enumerate(as_completed(futures)):
        if (i + 1) % 100 == 0:
            with stats_lock:
                print(f"[{i+1}/{len(futures)}] Crashes:{stats['crashes']} Bugs:{stats['miscompiles']}")
    
    executor.shutdown()
    
    # Summary
    print(f"\n{'='*60}")
    print("RETEST COMPLETE")
    print(f"{'='*60}")
    print(f"Tested: {len(ll_files)} files × {len(passes)} passes")
    print(f"Crashes: {stats['crashes']}")
    print(f"Miscompiles: {stats['miscompiles']}")
    
    # Save stats
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))