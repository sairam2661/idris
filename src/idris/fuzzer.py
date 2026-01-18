"""
LLVM IR Fuzzer using Language Models

Main fuzzing loop that generates IR, validates it, and tracks bugs.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import json

from vllm import LLM, SamplingParams

from idris.config import get_config
from idris.prompts import collect_seeds, PromptGenerator
from idris.scoring import score_seeds, get_scoring_stats
from idris.utils import complete_function, extract_logprobs, save_perplexity_analysis
from idris.validation import validation_worker


def run_idris():
    """Main fuzzing loop"""
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
    
    # Optional settings with defaults
    mutation_prob = cfg["fuzzer"].get("mutation_prob", 0.4)
    vector_ratio = cfg["fuzzer"].get("vector_ratio", 0.3)
    truncate_min = cfg["fuzzer"].get("truncate_min", 0.3)
    truncate_max = cfg["fuzzer"].get("truncate_max", 0.8)
    
    # New: perplexity-based seed selection settings
    score_seeds_enabled = cfg["fuzzer"].get("score_seeds", True)
    weird_ratio = cfg["fuzzer"].get("weird_ratio", 0.3)
    prefer_weird_prob = cfg["fuzzer"].get("prefer_weird_prob", 0.5)
    
    # Setup output directories
    output_dir.mkdir(exist_ok=True)
    bugs_dir = output_dir / "bugs"
    bugs_dir.mkdir(exist_ok=True)
    valid_dir = output_dir / "valid"
    valid_dir.mkdir(exist_ok=True)
    fp_dir = output_dir / "false_positives"
    fp_dir.mkdir(exist_ok=True)
    
    # Collect seeds using new parser
    print("Collecting and parsing seed functions...")
    seeds = collect_seeds(llvm_test_dir, max_count=10000)
    print(f"Collected {len(seeds)} parsed seeds")
    
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
        print(f"  - Mean perplexity: {scoring_stats['mean_perplexity']:.2f}")
        print(f"  - Median perplexity: {scoring_stats['median_perplexity']:.2f}")
        
        # Save scoring stats
        (output_dir / "scoring_stats.json").write_text(json.dumps(scoring_stats, indent=2))
        
        # Save full scores for analysis
        scores_data = [
            {"name": s.name, "perplexity": ppl, "length": len(s.raw_text)}
            for s, ppl in seed_scores
        ]
        (output_dir / "seed_scores.json").write_text(json.dumps(scores_data, indent=2))
    
    # Create prompt generator
    prompt_gen = PromptGenerator(
        seeds=seeds,
        seed_scores=seed_scores,
        weird_ratio=weird_ratio,
        prefer_weird_prob=prefer_weird_prob,
        mutation_prob=mutation_prob,
        truncate_min=truncate_min,
        truncate_max=truncate_max,
    )
    
    # Log seed statistics
    seed_stats = prompt_gen.stats()
    print(f"Seed statistics:")
    print(f"  - Total functions: {seed_stats['total_functions']}")
    print(f"  - Clusters: {seed_stats['num_clusters']}")
    print(f"  - Vector functions: {seed_stats['vector_seeds']}")
    print(f"  - Weird seeds: {seed_stats['weird_seeds']}")
    print(f"  - Normal seeds: {seed_stats['normal_seeds']}")
    print(f"  - Functions with CHECKs: {seed_stats['functions_with_checks']}")
    
    (output_dir / "seed_stats.json").write_text(json.dumps(seed_stats, indent=2))
    
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
        # Track by mutation strategy
        "by_strategy": {},
        # New: track by seed weirdness
        "bugs_from_weird_seeds": 0,
        "bugs_from_normal_seeds": 0,
        "generated_from_weird_seeds": 0,
        "generated_from_normal_seeds": 0,
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
        
        # Generate batch of prompts using new generator
        batch = prompt_gen.generate_batch(
            batch_size=batch_size,
            vector_ratio=vector_ratio,
        )
        
        prompts = [p for p, _ in batch]
        metadata_list = [m for _, m in batch]
        
        # Generate completions
        outputs = llm.generate(prompts, sampling_params)
        
        with stats_lock:
            stats["generated"] += len(outputs)
            for meta in metadata_list:
                if meta.get("is_weird_seed"):
                    stats["generated_from_weird_seeds"] += 1
                else:
                    stats["generated_from_normal_seeds"] += 1
        
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
            ppl_info["mutation_strategy"] = metadata["strategy"]
            ppl_info["seed_has_vectors"] = metadata["seed_has_vectors"]
            ppl_info["is_weird_seed"] = metadata.get("is_weird_seed", False)
            ppl_info["seed_perplexity"] = metadata.get("seed_perplexity")
            
            # Submit for validation
            future = executor.submit(
                validation_worker,
                (ir, prompt, metadata["strategy"], ppl_info),
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
            weird_bugs = stats.get('bugs_from_weird_seeds', 0)
            normal_bugs = stats.get('bugs_from_normal_seeds', 0)
            print(f"[{iteration+1}/{num_iterations}] {iter_time:.1f}s | "
                  f"Gen:{stats['generated']} Valid:{stats['valid']} Opt:{stats['optimized']} "
                  f"Correct:{stats['correct']} Bugs:{stats['bugs']} (W:{weird_bugs}/N:{normal_bugs}) "
                  f"Crashes:{crashes} FP:{stats['false_positives']} Timeout:{stats['timeout']} "
                  f"Pending:{len(pending_futures)}")
        
        # Periodic saves
        if iteration % 10 == 0:
            with stats_lock:
                stats_copy = dict(stats)
            stats_copy["elapsed_seconds"] = elapsed
            stats_copy["iteration"] = iteration
            (output_dir / "stats.json").write_text(json.dumps(stats_copy, indent=2))
            
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
    (output_dir / "stats.json").write_text(json.dumps(stats_copy, indent=2))
    
    with ppl_lock:
        save_perplexity_analysis(perplexity_log, output_dir)
    
    # Print final summary with weird vs normal breakdown
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total bugs found: {stats['bugs']}")
    print(f"  - From weird seeds: {stats['bugs_from_weird_seeds']}")
    print(f"  - From normal seeds: {stats['bugs_from_normal_seeds']}")
    print(f"\nGenerated programs:")
    print(f"  - From weird seeds: {stats['generated_from_weird_seeds']}")
    print(f"  - From normal seeds: {stats['generated_from_normal_seeds']}")
    
    if stats['generated_from_weird_seeds'] > 0:
        weird_rate = stats['bugs_from_weird_seeds'] / stats['generated_from_weird_seeds'] * 100
        print(f"\nBug rate (weird seeds): {weird_rate:.4f}%")
    if stats['generated_from_normal_seeds'] > 0:
        normal_rate = stats['bugs_from_normal_seeds'] / stats['generated_from_normal_seeds'] * 100
        print(f"Bug rate (normal seeds): {normal_rate:.4f}%")
    
    print(f"\nCrashes: {stats.get('crashes_verify', 0) + stats.get('crashes_opt', 0)}")
    print(f"False positives filtered: {stats['false_positives']}")