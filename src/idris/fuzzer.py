"""
LLVM IR Fuzzer using Language Models - Pattern-Guided Version

This version removes FileCheck mutation complexity and focuses on
generating prompts from signatures + semantic hints.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import json

from vllm import LLM, SamplingParams

from idris.config import get_config
from idris.parser import parse_functions_from_directory
from idris.prompts import (
    PatternGuidedGenerator,
    PromptStrategy,
    collect_seeds,
)
from idris.scoring import score_seeds, get_scoring_stats
from idris.utils import complete_function, extract_logprobs, save_perplexity_analysis
from idris.validation import validation_worker


def run_idris():
    """Main fuzzing loop with pattern-guided generation."""
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
    
    # Strategy settings
    vector_ratio = cfg["fuzzer"].get("vector_ratio", 0.3)
    scalable_ratio = cfg["fuzzer"].get("scalable_ratio", 0.2)
    
    # Perplexity-based seed selection
    score_seeds_enabled = cfg["fuzzer"].get("score_seeds", True)
    weird_ratio = cfg["fuzzer"].get("weird_ratio", 0.3)
    prefer_weird_prob = cfg["fuzzer"].get("prefer_weird_prob", 0.5)
    
    # Optional: force a specific strategy for A/B testing
    # Set to None to use weighted random selection
    force_strategy_name = cfg["fuzzer"].get("force_strategy", None)
    force_strategy = None
    if force_strategy_name:
        try:
            force_strategy = PromptStrategy(force_strategy_name)
            print(f"Forcing strategy: {force_strategy.value}")
        except ValueError:
            print(f"Unknown strategy '{force_strategy_name}', using random")
    
    # Setup output directories
    output_dir.mkdir(exist_ok=True)
    bugs_dir = output_dir / "bugs"
    bugs_dir.mkdir(exist_ok=True)
    valid_dir = output_dir / "valid"
    valid_dir.mkdir(exist_ok=True)
    fp_dir = output_dir / "false_positives"
    fp_dir.mkdir(exist_ok=True)
    
    # Collect seeds
    print("Collecting seed functions...")
    seeds = collect_seeds(llvm_test_dir, max_count=10000)
    print(f"Collected {len(seeds)} seeds")
    
    # Load model
    print(f"Loading {model_name}...")
    llm = LLM(model=model_name, trust_remote_code=True)
    
    # Score seeds by perplexity
    seed_scores = None
    if score_seeds_enabled:
        print("Scoring seeds by perplexity...")
        seed_scores = score_seeds(llm, seeds, batch_size=64, verbose=True)
        
        scoring_stats = get_scoring_stats(seed_scores)
        print(f"Scoring complete:")
        print(f"  - Scored: {scoring_stats['num_scored']} seeds")
        print(f"  - Perplexity range: {scoring_stats['min_perplexity']:.2f} - {scoring_stats['max_perplexity']:.2f}")
        print(f"  - Mean: {scoring_stats['mean_perplexity']:.2f}")
        
        (output_dir / "scoring_stats.json").write_text(json.dumps(scoring_stats, indent=2))
    
    # Create pattern-guided generator
    generator = PatternGuidedGenerator(
        seeds=seeds,
        seed_scores=seed_scores,
        weird_ratio=weird_ratio,
        prefer_weird_prob=prefer_weird_prob,
    )
    
    # Log generator stats
    gen_stats = generator.stats()
    print(f"\nGenerator statistics:")
    print(f"  - Total seeds: {gen_stats['total_seeds']}")
    print(f"  - Vector seeds: {gen_stats['vector_seeds']}")
    print(f"  - Scalable vector seeds: {gen_stats['scalable_vector_seeds']}")
    print(f"  - Weird seeds: {gen_stats['weird_seeds']}")
    print(f"  - Normal seeds: {gen_stats['normal_seeds']}")
    
    (output_dir / "generator_stats.json").write_text(json.dumps(gen_stats, indent=2))
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        presence_penalty=0.15,
        logprobs=1,
    )
    
    # Statistics tracking
    stats = {
        "generated": 0,
        "valid": 0,
        "optimized": 0,
        "correct": 0,
        "bugs": 0,
        "timeout": 0,
        "false_positives": 0,
        "crashes_verify": 0,
        "crashes_opt": 0,
        # Track by strategy
        "by_strategy": {s.value: {"generated": 0, "valid": 0, "bugs": 0, "crashes": 0} 
                       for s in PromptStrategy},
        # Track by seed type
        "bugs_from_weird_seeds": 0,
        "bugs_from_normal_seeds": 0,
        "generated_from_weird_seeds": 0,
        "generated_from_normal_seeds": 0,
        # Track by vector type
        "bugs_from_scalable_vectors": 0,
        "generated_with_scalable_vectors": 0,
    }
    stats_lock = threading.Lock()
    
    perplexity_log = []
    ppl_lock = threading.Lock()
    
    # Thread pool for validation
    executor = ThreadPoolExecutor(max_workers=num_workers)
    pending_futures = []
    
    start_time = time.time()
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Generate batch of prompts
        batch = generator.generate_batch(
            batch_size=batch_size,
            vector_ratio=vector_ratio,
            scalable_ratio=scalable_ratio,
            strategy=force_strategy,  # None = random weighted selection
        )
        
        prompts = [p for p, _ in batch]
        metadata_list = [m for _, m in batch]
        
        # Generate completions
        outputs = llm.generate(prompts, sampling_params)
        
        # Update generation stats
        with stats_lock:
            stats["generated"] += len(outputs)
            for meta in metadata_list:
                strategy = meta.get("strategy", "unknown")
                if strategy in stats["by_strategy"]:
                    stats["by_strategy"][strategy]["generated"] += 1
                
                if meta.get("is_weird_seed"):
                    stats["generated_from_weird_seeds"] += 1
                else:
                    stats["generated_from_normal_seeds"] += 1
                
                if meta.get("seed_has_scalable_vectors"):
                    stats["generated_with_scalable_vectors"] += 1
        
        # Process completions
        for idx, output in enumerate(outputs):
            prompt = prompts[idx]
            metadata = metadata_list[idx]
            completion = output.outputs[0].text
            
            # Skip very short completions
            if len(completion) < 30:
                continue
            
            # Try to complete the function
            ir = complete_function(prompt, completion)
            if ir is None:
                continue
            
            # Extract perplexity info
            ppl_info = extract_logprobs(output)
            ppl_info["strategy"] = metadata.get("strategy")
            ppl_info["seed_has_vectors"] = metadata.get("seed_has_vectors", False)
            ppl_info["seed_has_scalable_vectors"] = metadata.get("seed_has_scalable_vectors", False)
            ppl_info["is_weird_seed"] = metadata.get("is_weird_seed", False)
            ppl_info["seed_perplexity"] = metadata.get("seed_perplexity")
            ppl_info["pattern_name"] = metadata.get("pattern_name")
            
            # Submit for validation
            future = executor.submit(
                validation_worker,
                (ir, prompt, metadata.get("strategy", "unknown"), ppl_info),
                bugs_dir, valid_dir, fp_dir, stats, stats_lock,
                perplexity_log, ppl_lock
            )
            pending_futures.append(future)
        
        # Clean up completed futures
        done_futures = [f for f in pending_futures if f.done()]
        pending_futures = [f for f in pending_futures if not f.done()]
        
        # Progress update
        iter_time = time.time() - iter_start
        elapsed = time.time() - start_time
        
        with stats_lock:
            crashes = stats.get('crashes_verify', 0) + stats.get('crashes_opt', 0)
            
            # Strategy breakdown
            strategy_bugs = {k: v["bugs"] for k, v in stats["by_strategy"].items() if v["bugs"] > 0}
            strategy_str = " ".join([f"{k}:{v}" for k, v in strategy_bugs.items()]) if strategy_bugs else ""
            
            print(f"[{iteration+1}/{num_iterations}] {iter_time:.1f}s | "
                  f"Gen:{stats['generated']} Valid:{stats['valid']} "
                  f"Bugs:{stats['bugs']} Crashes:{crashes} "
                  f"Pending:{len(pending_futures)}")
            if strategy_str:
                print(f"  Bugs by strategy: {strategy_str}")
        
        # Periodic saves
        if iteration % 10 == 0:
            with stats_lock:
                stats_copy = dict(stats)
            stats_copy["elapsed_seconds"] = elapsed
            stats_copy["iteration"] = iteration
            (output_dir / "stats.json").write_text(json.dumps(stats_copy, indent=2, default=str))
            
            with ppl_lock:
                save_perplexity_analysis(list(perplexity_log), output_dir)
    
    # Wait for remaining validations
    print("Waiting for pending validations...")
    for f in as_completed(pending_futures):
        pass
    
    executor.shutdown()
    
    # Final save
    with stats_lock:
        stats_copy = dict(stats)
    stats_copy["elapsed_seconds"] = time.time() - start_time
    stats_copy["completed"] = True
    (output_dir / "stats.json").write_text(json.dumps(stats_copy, indent=2, default=str))
    
    with ppl_lock:
        save_perplexity_analysis(perplexity_log, output_dir)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total bugs found: {stats['bugs']}")
    print(f"Total crashes: {stats.get('crashes_verify', 0) + stats.get('crashes_opt', 0)}")
    
    print(f"\nBugs by strategy:")
    for strategy, data in stats["by_strategy"].items():
        if data["generated"] > 0:
            bug_rate = data["bugs"] / data["generated"] * 100 if data["generated"] > 0 else 0
            print(f"  {strategy}: {data['bugs']} bugs / {data['generated']} generated ({bug_rate:.3f}%)")
    
    print(f"\nBugs by seed type:")
    print(f"  From weird seeds: {stats['bugs_from_weird_seeds']}")
    print(f"  From normal seeds: {stats['bugs_from_normal_seeds']}")
    
    if stats.get('generated_with_scalable_vectors', 0) > 0:
        print(f"\nScalable vector stats:")
        print(f"  Generated: {stats['generated_with_scalable_vectors']}")
        print(f"  Bugs: {stats.get('bugs_from_scalable_vectors', 0)}")


def run_ab_test():
    """
    Run an A/B test comparing different strategies.
    
    This runs the fuzzer multiple times with different forced strategies
    and compares the results.
    """
    cfg = get_config()
    output_base = Path(cfg["paths"]["output_dir"])
    
    strategies_to_test = [
        PromptStrategy.SIGNATURE_ONLY,
        PromptStrategy.SIGNATURE_WITH_ENTRY,
        PromptStrategy.PATTERN_GUIDED,
        PromptStrategy.TYPE_NOVEL,
    ]
    
    results = {}
    
    for strategy in strategies_to_test:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy.value}")
        print(f"{'='*60}")
        
        # Update config to force this strategy
        cfg["fuzzer"]["force_strategy"] = strategy.value
        cfg["paths"]["output_dir"] = str(output_base / f"ab_test_{strategy.value}")
        
        # Run fuzzer
        run_idris()
        
        # Load results
        stats_file = Path(cfg["paths"]["output_dir"]) / "stats.json"
        if stats_file.exists():
            results[strategy.value] = json.loads(stats_file.read_text())
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"A/B TEST RESULTS")
    print(f"{'='*60}")
    
    for strategy, data in results.items():
        generated = data.get("generated", 0)
        bugs = data.get("bugs", 0)
        crashes = data.get("crashes_verify", 0) + data.get("crashes_opt", 0)
        rate = bugs / generated * 100 if generated > 0 else 0
        
        print(f"{strategy}:")
        print(f"  Generated: {generated}")
        print(f"  Bugs: {bugs} ({rate:.4f}%)")
        print(f"  Crashes: {crashes}")
        print()