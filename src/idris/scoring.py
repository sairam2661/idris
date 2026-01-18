import math
from typing import Optional
from vllm import LLM, SamplingParams

from idris.parser import ParsedFunction


def compute_perplexity_from_logprobs(logprobs_list: list[float]) -> Optional[float]:
    """Compute perplexity from a list of log probabilities."""
    if not logprobs_list or len(logprobs_list) == 0:
        return None
    
    valid_logprobs = [lp for lp in logprobs_list if lp is not None]
    if not valid_logprobs:
        return None
    
    avg_neg_logprob = -sum(valid_logprobs) / len(valid_logprobs)
    perplexity = math.exp(avg_neg_logprob)
    return perplexity


def score_seeds(
    llm: LLM, 
    seeds: list[ParsedFunction], 
    batch_size: int = 32,
    verbose: bool = True
) -> list[tuple[ParsedFunction, float]]:
    """
    Score seeds by perplexity under the language model.
    
    Higher perplexity = lower likelihood = "weirder" seed.
    
    Args:
        llm: The vLLM model instance
        seeds: List of parsed seed functions
        batch_size: Batch size for scoring
        verbose: Print progress
    
    Returns:
        List of (seed, perplexity) tuples, sorted by perplexity descending (weirdest first)
    """
    scoring_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        prompt_logprobs=1,
    )
    
    scored = []
    num_batches = (len(seeds) + batch_size - 1) // batch_size
    
    for i in range(0, len(seeds), batch_size):
        batch = seeds[i:i + batch_size]
        prompts = [s.raw_text for s in batch]
        
        outputs = llm.generate(prompts, scoring_params)
        
        for seed, output in zip(batch, outputs):
            if output.prompt_logprobs is not None:
                logprobs = []
                for token_logprob in output.prompt_logprobs:
                    if token_logprob is not None:
                        for _, logprob_obj in token_logprob.items():
                            logprobs.append(logprob_obj.logprob)
                            break
                
                ppl = compute_perplexity_from_logprobs(logprobs)
                if ppl is not None:
                    scored.append((seed, ppl))
        
        if verbose:
            batch_num = i // batch_size + 1
            print(f"  Scored batch {batch_num}/{num_batches} ({len(scored)} seeds scored)")
    
    # Sort by perplexity descending (weirdest first)
    scored.sort(key=lambda x: x[1], reverse=True)
    
    return scored


def get_scoring_stats(scored_seeds: list[tuple[ParsedFunction, float]]) -> dict:
    """Get statistics about scored seeds."""
    if not scored_seeds:
        return {}
    
    perplexities = [ppl for _, ppl in scored_seeds]
    
    return {
        "num_scored": len(scored_seeds),
        "max_perplexity": max(perplexities),
        "min_perplexity": min(perplexities),
        "mean_perplexity": sum(perplexities) / len(perplexities),
        "median_perplexity": sorted(perplexities)[len(perplexities) // 2],
    }