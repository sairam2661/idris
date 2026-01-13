import os
import tempfile
import subprocess
import json 
from datetime import datetime
from idris.config import get_config
from pathlib import Path

def verify_ir(ir, opt_path):
	with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
		f.write(ir)
		path = f.name
	try:
		result = subprocess.run([opt_path, '-passes=verify', '-S', path], capture_output=True, timeout=5)
		return result.returncode == 0
	except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
			return False
	except FileNotFoundError:
			print(f"Error: {opt_path} not found.")
			return False
	finally:
		if os.path.exists(path):
			os.unlink(path)

def optimize_ir(ir, opt_path):
	with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f:
		f.write(ir)
		path = f.name
	try:
		result = subprocess.run([opt_path, '-passes=default<O2>', '-S', path], capture_output=True, text=True, timeout=30)
		if result.returncode == 0:
			return True, result.stdout
		return False, None
	except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
			return False, None
	except FileNotFoundError:
			print(f"Error: {opt_path} not found.")
			return False, None
	finally:
		if os.path.exists(path):
			os.unlink(path)

def is_known_false_positive(alive_output, tgt_ir):
	# Issue #1202: initializes attribute not handled correctly
	if "initializes(" in tgt_ir:
		return "initializes_attr"
	
	# TODO: Add more patterns
	
	return None


def check_alive(src, tgt, alive_tv_path):
	with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f1:
		f1.write(src)
		src_path = f1.name
	with tempfile.NamedTemporaryFile(mode='w', suffix='.ll', delete=False) as f2:
		f2.write(tgt)
		tgt_path = f2.name
	try:
		result = subprocess.run(
			[alive_tv_path, src_path, tgt_path, "--disable-undef-input"],
			capture_output=True, text=True, timeout=40
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
		return {"correct": False, "incorrect": False, "false_positive": False, 
				"false_positive_reason": None, "timeout": True, "output": "timeout"}
	except Exception as e:
		return {"correct": False, "incorrect": False, "false_positive": False,
				"false_positive_reason": None, "timeout": False, "output": str(e)}
	finally:
		os.unlink(src_path)
		os.unlink(tgt_path)


def validation_worker(item, bugs_dir, valid_dir, fp_dir, stats, stats_lock, perplexity_log, ppl_lock):
	cfg = get_config()
	opt_path = Path(cfg["paths"]["opt"])
	alive_tv_path = Path(cfg["paths"]["alive_tv"])
 
	ir, prompt, prompt_type, ppl_info = item
  
	record = {
		"prompt_type": prompt_type,
		"ir_length": len(ir),
		"perplexity": ppl_info.get("perplexity"),
		"mean_logprob": ppl_info.get("mean_logprob"),
		"min_logprob": ppl_info.get("min_logprob"),
		"num_tokens": ppl_info.get("num_tokens"),
		"outcome": None,
	}
	
	if not verify_ir(ir, opt_path):
		record["outcome"] = "invalid"
		with ppl_lock:
			perplexity_log.append(record)
		return "invalid"
	
	valid_id = None
	with stats_lock:
		stats["valid"] += 1
		valid_id = stats["valid"]
		
	(valid_dir / f"valid_{valid_id:06d}.ll").write_text(ir)
	
	opt_ok, optimized = optimize_ir(ir, opt_path)
	if not opt_ok:
		record["outcome"] = "opt_failed"
		with ppl_lock:
			perplexity_log.append(record)
		return "opt_failed"
	
	with stats_lock:
		stats["optimized"] += 1
	
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
		}))
		
		ppl_str = f" PPL: {ppl_info.get('perplexity'):.2f}" if ppl_info.get('perplexity') else ""
		print(f"\n[BUG {bug_id}] Found miscompilation!{ppl_str}")
		print(f"Source:\n{ir[:200]}...")
		
		record["outcome"] = "bug"
		with ppl_lock:
			perplexity_log.append(record)
		return "bug"
	
	elif result["timeout"]:
		with stats_lock:
			stats["timeout"] += 1
		record["outcome"] = "timeout"
		with ppl_lock:
			perplexity_log.append(record)
		return "timeout"
	
	record["outcome"] = "unknown"
	with ppl_lock:
		perplexity_log.append(record)
	return "unknown"

