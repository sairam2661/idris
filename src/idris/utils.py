import random
import math 
import json
import csv 

def truncate_randomly(func):
    ratio = random.uniform(0.3, 0.8)
    cut_point = int(len(func) * ratio)
    newline_pos = func.rfind('\n', 0, cut_point)
    if newline_pos > 50:
        return func[:newline_pos + 1]
    return func[:cut_point]

def complete_function(prompt, completion):
    full = prompt + completion
    brace_count = 0
    
    for i, char in enumerate(full):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return full[:i + 1]
    
    return None

def compute_perplexity_from_logprobs(logprobs_list):
    if not logprobs_list or len(logprobs_list) == 0:
        return None
    
    valid_logprobs = [lp for lp in logprobs_list if lp is not None]
    if not valid_logprobs:
        return None
    
    avg_neg_logprob = -sum(valid_logprobs) / len(valid_logprobs)
    perplexity = math.exp(avg_neg_logprob)
    return perplexity

def extract_logprobs(output):
    logprobs_list = []
    
    if output.outputs[0].logprobs is not None:
        for token_logprob in output.outputs[0].logprobs:
            if token_logprob:
                for _, logprob_obj in token_logprob.items():
                    logprobs_list.append(logprob_obj.logprob)
                    break
    
    perplexity = compute_perplexity_from_logprobs(logprobs_list)
    
    return {
        "logprobs": logprobs_list,
        "perplexity": perplexity,
        "num_tokens": len(logprobs_list),
        "sum_logprobs": sum(logprobs_list) if logprobs_list else None,
        "mean_logprob": sum(logprobs_list) / len(logprobs_list) if logprobs_list else None,
        "min_logprob": min(logprobs_list) if logprobs_list else None,
        "max_logprob": max(logprobs_list) if logprobs_list else None,
    }

def save_perplexity_analysis(perplexity_log, output_dir):
    if not perplexity_log:
        return
    
    with open(output_dir / "perplexity_log.json", "w") as f:
        json.dump(perplexity_log, f, indent=2)
    
    fieldnames = ["outcome", "perplexity", "mean_logprob", "min_logprob", "num_tokens", "ir_length", "prompt_type"]
    with open(output_dir / "perplexity_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(perplexity_log)
    
    outcomes = {}
    for record in perplexity_log:
        outcome = record["outcome"]
        if outcome not in outcomes:
            outcomes[outcome] = {"perplexities": [], "mean_logprobs": []}
        if record["perplexity"] is not None:
            outcomes[outcome]["perplexities"].append(record["perplexity"])
        if record["mean_logprob"] is not None:
            outcomes[outcome]["mean_logprobs"].append(record["mean_logprob"])
    
    summary = {}
    for outcome, data in outcomes.items():
        ppls = data["perplexities"]
        mlps = data["mean_logprobs"]
        summary[outcome] = {
            "count": len(ppls),
            "perplexity_mean": sum(ppls) / len(ppls) if ppls else None,
            "perplexity_median": sorted(ppls)[len(ppls)//2] if ppls else None,
            "perplexity_min": min(ppls) if ppls else None,
            "perplexity_max": max(ppls) if ppls else None,
            "mean_logprob_avg": sum(mlps) / len(mlps) if mlps else None,
        }
    
    with open(output_dir / "perplexity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)