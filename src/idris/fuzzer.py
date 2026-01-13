
from pathlib import Path
from idris.config import get_config
from idris.prompts import extract_functions
from idris.utils import truncate_randomly, complete_function, extract_logprobs, save_perplexity_analysis
from idris.validation import validation_worker
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random 
import json 

def run_idris():
	cfg = get_config()
	llvm_test_dir = Path(cfg["paths"]["llvm_test_dir"])
	output_dir = Path(cfg["paths"]["output_dir"])
	batch_size = cfg["fuzzer"]["batch_size"]
	num_iterations = cfg["fuzzer"]["num_iterations"]
	temperature = cfg["fuzzer"]["temperature"]
	max_tokens = cfg["fuzzer"]["max_tokens"]
	num_workers = cfg["fuzzer"]["num_workers"]
	model_name = cfg["fuzzer"]["model"]
		
	output_dir.mkdir(exist_ok=True)
	bugs_dir = output_dir / "bugs"
	bugs_dir.mkdir(exist_ok=True)
	valid_dir = output_dir / "valid"
	valid_dir.mkdir(exist_ok=True)
	fp_dir = output_dir / "false_positives"
	fp_dir.mkdir(exist_ok=True)
	
	print("Collecting seed functions...")
	seed_functions = extract_functions(llvm_test_dir, max_count=10000)
	print(f"Collected {len(seed_functions)} seeds")
	
	print(f"Loading {model_name}...")
	llm = LLM(model=model_name, trust_remote_code=True)
	
	sampling_params = SamplingParams(
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=0.9,
		presence_penalty=0.15,
		logprobs=1,
	)
	
	stats = {"generated": 0, "valid": 0, "optimized": 0, "correct": 0, 
			 "bugs": 0, "timeout": 0, "false_positives": 0}
	stats_lock = threading.Lock()
	
	perplexity_log = []
	ppl_lock = threading.Lock()
	
	executor = ThreadPoolExecutor(max_workers=num_workers)
	pending_futures = []
	
	start_time = time.time()
	
	for iteration in range(num_iterations):
		iter_start = time.time()
		
		prompts = []
		prompt_types = []
		
		for _ in range(batch_size):
			func = random.choice(seed_functions)
			prompt = truncate_randomly(func)
			prompts.append(prompt)
			prompt_types.append("seed")
		
		outputs = llm.generate(prompts, sampling_params)
		
		with stats_lock:
			stats["generated"] += len(outputs)
		
		for idx, output in enumerate(outputs):
			prompt = prompts[idx]
			prompt_type = prompt_types[idx]
			completion = output.outputs[0].text
			
			if len(completion) < 30:
				continue
			
			ir = complete_function(prompt, completion)
			if ir is None:
				continue
			
			ppl_info = extract_logprobs(output)
			
			future = executor.submit(
				validation_worker, 
				(ir, prompt, prompt_type, ppl_info), 
				bugs_dir, valid_dir, fp_dir, stats, stats_lock,
				perplexity_log, ppl_lock
			)
			pending_futures.append(future)
		
		done_futures = [f for f in pending_futures if f.done()]
		pending_futures = [f for f in pending_futures if not f.done()]
		
		iter_time = time.time() - iter_start
		elapsed = time.time() - start_time
		
		with stats_lock:
			print(f"[{iteration+1}/{num_iterations}] {iter_time:.1f}s | "
				  f"Gen:{stats['generated']} Valid:{stats['valid']} Opt:{stats['optimized']} "
				  f"Correct:{stats['correct']} Bugs:{stats['bugs']} FP:{stats['false_positives']} "
				  f"Timeout:{stats['timeout']} Pending:{len(pending_futures)}")
		
		if iteration % 10 == 0:
			with stats_lock:
				stats_copy = dict(stats)
			stats_copy["elapsed_seconds"] = elapsed
			stats_copy["iteration"] = iteration
			(output_dir / "stats.json").write_text(json.dumps(stats_copy, indent=2))
			
			with ppl_lock:
				save_perplexity_analysis(list(perplexity_log), output_dir)
	
	print("Waiting for pending validations...")
	for f in as_completed(pending_futures):
		pass
	
	executor.shutdown()
	
	with stats_lock:
		stats_copy = dict(stats)
	stats_copy["elapsed_seconds"] = time.time() - start_time
	stats_copy["completed"] = True
	(output_dir / "stats.json").write_text(json.dumps(stats_copy, indent=2))
	
	with ppl_lock:
		save_perplexity_analysis(perplexity_log, output_dir)
	
	print(f"\nDone! Total bugs found: {stats['bugs']} (filtered {stats['false_positives']} known false positives)")

