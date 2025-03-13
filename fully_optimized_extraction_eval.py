#!/usr/bin/env python
"""
Fully Optimized Extraction Evaluation.

This script combines Phase 1 and Phase 2 optimizations into a cohesive 
evaluation solution with optimal performance.

Optimizations included:
- Phase 1:
  - Parallel API Request Processing
  - Response Caching
  - Adaptive Rate Limiting
- Phase 2:
  - Tiered Prompt System
  - Optimized JSON Parsing
  - Metrics Calculation Optimization

Usage:
python fully_optimized_extraction_eval.py --prompt-file prompt_template.txt --eval-data eval_data.json --tier auto
"""

import os
import time
import json
import click
from pathlib import Path
from typing import Dict, List, Any

from sentence_extraction import (
    # Core functionality
    AnthropicAdapter, 
    OpenAIAdapter,
    GeminiAdapter,
    DeepSeekAdapter,
    load_data,
    format_prompt,
    create_output_directory,
    save_results,
    CLAUDE_MODELS,
    OPENAI_MODELS,
    GEMINI_MODELS,
    DEEPSEEK_MODELS,
    
    # Phase 1 optimizations
    ResponseCache,
    ModelRateLimiter,
    process_data_parallel,
    
    # Phase 2 optimizations
    extract_json_efficiently,
    TieredPromptSystem,
    calculate_metrics_optimized
)

def process_with_optimizations(
    client, 
    prompt_system: TieredPromptSystem,
    item: Dict[str, Any],
    model: str,
    tier: str = "auto",
    temperature: float = 0.0,
    rate_limiter = None,
    cache = None
) -> Dict[str, Any]:
    """
    Process a single item with full optimizations from Phase 1 and Phase 2.
    
    Args:
        client: Model client adapter
        prompt_system: Tiered prompt system
        item: Item to process
        model: Model ID
        tier: Prompt tier to use
        temperature: Temperature for model responses
        rate_limiter: Optional rate limiter
        cache: Optional cache
        
    Returns:
        Dictionary with processing results
    """
    sentence = item["sentence"]
    expected = item["extracted_information"]
    
    # Select appropriate tier if auto
    if tier == "auto":
        selected_tier = prompt_system.get_recommended_tier(sentence)
        complexity_analysis = prompt_system.analyze_sentence_complexity(sentence)
    else:
        selected_tier = tier
        complexity_analysis = None
    
    # Get prompt for the selected tier
    prompt_template = prompt_system.get_prompt(selected_tier)
    
    # Format prompt with the sentence
    prompt = format_prompt(prompt_template, sentence)
    
    # Apply rate limiting if provided
    if rate_limiter:
        rate_limiter.wait()
    
    # Check cache first if provided
    cached_response = None
    if cache:
        cache_key = f"{selected_tier}:{prompt}"
        cached_response = cache.get(prompt, sentence, model)
    
    start_time = time.time()
    try:
        if cached_response:
            # Use cached response
            response_text = cached_response
            api_call = False
        else:
            # Call model API
            response_text = client(prompt, model, temperature)
            api_call = True
            # Cache the response if cache is available
            if cache:
                cache.set(prompt, sentence, model, response_text)
        
        # Extract JSON using optimized extraction
        extracted = extract_json_efficiently(response_text)
        
        # Return results
        processing_time = time.time() - start_time
        return {
            "sentence": sentence,
            "expected": expected,
            "extracted": extracted,
            "full_response": response_text,
            "cached": cached_response is not None,
            "api_call": api_call,
            "selected_tier": selected_tier,
            "complexity_analysis": complexity_analysis,
            "processing_time": processing_time,
            "token_estimate": prompt_system.get_token_estimates().get(selected_tier, 0)
        }
    except Exception as e:
        processing_time = time.time() - start_time
        return {
            "sentence": sentence,
            "expected": expected,
            "extracted": {},
            "error": str(e),
            "full_response": "",
            "cached": False,
            "api_call": api_call if 'api_call' in locals() else False,
            "selected_tier": selected_tier,
            "complexity_analysis": complexity_analysis,
            "processing_time": processing_time
        }

@click.command()
@click.option("--prompt-file", required=True, type=click.Path(exists=True), help="Path to the base prompt template file")
@click.option("--eval-data", required=True, type=click.Path(exists=True), help="Path to the evaluation data JSON file")
@click.option("--api-key", help="API key (or set via environment variables)")
@click.option("--model", 
    type=click.Choice(["opus", "sonnet", "haiku", "gpt4o", "gpt4", "gpt35", 
                      "pro", "pro-vision", "ultra", "coder", "chat"], 
                      case_sensitive=False), 
    default="sonnet", 
    help="Model to use")
@click.option("--tier", type=click.Choice(["fast", "standard", "comprehensive", "auto"]), default="auto", help="Prompt tier to use")
@click.option("--max-sentences", type=int, default=None, help="Maximum number of sentences to evaluate")
@click.option("--parallel/--no-parallel", default=True, help="Enable parallel processing")
@click.option("--max-workers", type=int, default=8, help="Maximum number of parallel workers")
@click.option("--batch-size", type=int, default=10, help="Batch size for processing")
@click.option("--detailed-metrics/--fast-metrics", default=True, help="Use detailed metrics calculation")
@click.option("--cache/--no-cache", default=True, help="Enable response caching")
@click.option("--rate-limiting/--no-rate-limiting", default=True, help="Enable adaptive rate limiting")
def main(prompt_file, eval_data, api_key, model, tier, max_sentences, parallel, 
         max_workers, batch_size, detailed_metrics, cache, rate_limiting):
    """
    Run a fully optimized extraction evaluation with all Phase 1 and Phase 2 optimizations.
    """
    # Start timing
    start_time = time.time()
    
    # Determine model type
    is_claude = model.lower() in CLAUDE_MODELS
    is_openai = model.lower() in OPENAI_MODELS
    is_gemini = model.lower() in GEMINI_MODELS
    is_deepseek = model.lower() in DEEPSEEK_MODELS

    if not any([is_claude, is_openai, is_gemini, is_deepseek]):
        raise click.UsageError(f"Unknown model '{model}'. Choose from: {', '.join(list(CLAUDE_MODELS.keys()) + list(OPENAI_MODELS.keys()) + list(GEMINI_MODELS.keys()) + list(DEEPSEEK_MODELS.keys()))}")

    # Get API key and initialize client based on model type
    if is_claude:
        model_type = "claude"
        model_id = CLAUDE_MODELS.get(model.lower())
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise click.UsageError("Anthropic API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable")
        client = AnthropicAdapter(api_key)
    elif is_openai:
        model_type = "openai"
        model_id = OPENAI_MODELS.get(model.lower())
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise click.UsageError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY environment variable")
        client = OpenAIAdapter(api_key)
    elif is_gemini:
        model_type = "gemini"
        model_id = GEMINI_MODELS.get(model.lower())
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise click.UsageError("Google API key must be provided via --api-key or GOOGLE_API_KEY environment variable")
        client = GeminiAdapter(api_key)
    else:  # DeepSeek
        model_type = "deepseek"
        model_id = DEEPSEEK_MODELS.get(model.lower())
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise click.UsageError("DeepSeek API key must be provided via --api-key or DEEPSEEK_API_KEY environment variable")
        client = DeepSeekAdapter(api_key)
    
    # Load data
    prompt_template, sentences = load_data(prompt_file, eval_data, max_sentences)
    
    # Create output directory
    output_dir = create_output_directory(f"{model}-fully-optimized", prompt_file, f"{model_type}-{tier}")
    
    # Initialize tiered prompt system
    prompt_system = TieredPromptSystem(prompt_file)
    
    # Initialize cache and rate limiter if enabled
    response_cache = None
    if cache:
        response_cache = ResponseCache(cache_dir="./cache")
    
    rate_limiter_instance = None
    if rate_limiting:
        rate_limiter_instance = ModelRateLimiter().get_limiter(model_id)
    
    # Display configuration
    print(f"Fully Optimized Extraction Evaluation")
    print(f"====================================")
    print(f"Model: {model} ({model_id})")
    print(f"Prompt file: {prompt_file}")
    print(f"Eval data: {eval_data}")
    print(f"Sentences: {len(sentences)}")
    print(f"Phase 1 Optimizations:")
    print(f"  - Parallel Processing: {'Enabled' if parallel else 'Disabled'}")
    if parallel:
        print(f"    - Max workers: {max_workers}")
        print(f"    - Batch size: {batch_size}")
    print(f"  - Response Caching: {'Enabled' if cache else 'Disabled'}")
    print(f"  - Adaptive Rate Limiting: {'Enabled' if rate_limiting else 'Disabled'}")
    print(f"Phase 2 Optimizations:")
    print(f"  - Tiered Prompt System: Enabled (Tier: {tier})")
    print(f"  - Optimized JSON Parsing: Enabled")
    print(f"  - Metrics Calculation: {'Detailed' if detailed_metrics else 'Fast'}")
    print(f"Output directory: {output_dir}")
    print(f"====================================")
    
    # Prepare items for processing
    process_items = []
    for item in sentences:
        process_items.append({
            "sentence": item["sentence"],
            "extracted_information": item["extracted_information"]
        })
    
    # Process sentences
    processing_start = time.time()
    
    if parallel:
        # Process in batches with parallel processing
        results = []
        for i in range(0, len(process_items), batch_size):
            batch = process_items[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(process_items)-1)//batch_size + 1} with {len(batch)} sentences...")
            
            batch_results = process_data_parallel(
                batch,
                lambda item: process_with_optimizations(
                    client, prompt_system, item, model_id, tier, 
                    0.0, rate_limiter_instance, response_cache
                ),
                max_workers=max_workers
            )
            
            results.extend(batch_results)
            
            # Print batch stats
            batch_cache_hits = sum(1 for r in batch_results if r.get("cached", False))
            batch_cache_hit_rate = batch_cache_hits / len(batch) * 100
            print(f"  Cache hit rate: {batch_cache_hit_rate:.1f}%")
            
            # Show tier distribution
            tier_counts = {}
            for r in batch_results:
                selected_tier = r.get("selected_tier", "unknown")
                tier_counts[selected_tier] = tier_counts.get(selected_tier, 0) + 1
            
            print(f"  Tier distribution: ", end="")
            for t, count in tier_counts.items():
                print(f"{t}: {count} ({count/len(batch)*100:.1f}%), ", end="")
            print()
    else:
        # Sequential processing
        print(f"Processing {len(sentences)} sentences sequentially...")
        results = []
        for item in process_items:
            result = process_with_optimizations(
                client, prompt_system, item, model_id, tier, 
                0.0, rate_limiter_instance, response_cache
            )
            results.append(result)
    
    processing_time = time.time() - processing_start
    
    # Calculate metrics using optimized calculation
    metrics_start = time.time()
    metrics = calculate_metrics_optimized(results, detailed_metrics=detailed_metrics)
    metrics_time = time.time() - metrics_start
    
    # Print tier selection statistics
    tier_counts = {}
    for result in results:
        selected_tier = result.get("selected_tier", "unknown")
        tier_counts[selected_tier] = tier_counts.get(selected_tier, 0) + 1
    
    print("\nTier Selection Statistics:")
    for t, count in tier_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {t.capitalize()}: {count} sentences ({percentage:.1f}%)")
    
    # Print processing statistics
    print(f"\nPerformance Statistics:")
    print(f"  Total processing time: {processing_time:.2f} seconds")
    print(f"  Average time per sentence: {processing_time/len(sentences):.2f} seconds")
    print(f"  Processing rate: {len(sentences)/processing_time*60:.2f} sentences/minute")
    print(f"  Metrics calculation time: {metrics_time:.4f} seconds")
    
    # Print cache statistics if enabled
    if cache:
        cache_hits = sum(1 for r in results if r.get("cached", False))
        cache_misses = len(results) - cache_hits
        print(f"\nCache Statistics:")
        print(f"  Cache hits: {cache_hits}/{len(results)} ({cache_hits/len(results)*100:.1f}%)")
        print(f"  Cache misses: {cache_misses}/{len(results)} ({cache_misses/len(results)*100:.1f}%)")
    
    # Print accuracy metrics
    print(f"\nAccuracy Metrics:")
    print(f"  Overall accuracy: {metrics.get('overall_value_accuracy', 0)*100:.2f}%")
    print(f"  Successful extractions: {metrics['successful_extractions']}/{metrics['total_sentences']}")
    print(f"  Average score: {metrics['scores']['avg_score']*100:.2f}%")
    
    # Save results and metrics
    output_file = save_results(results, metrics, output_dir, model, eval_data, processing_time, prompt_file, model_type)
    
    # Save optimization-specific data
    with open(output_dir / "optimization_metrics.json", 'w') as f:
        json.dump({
            "configuration": {
                "model": model,
                "model_id": model_id,
                "tier": tier,
                "parallel": parallel,
                "max_workers": max_workers,
                "batch_size": batch_size,
                "detailed_metrics": detailed_metrics,
                "cache_enabled": cache,
                "rate_limiting_enabled": rate_limiting
            },
            "tier_statistics": {
                "counts": tier_counts,
                "percentages": {t: (count/len(results)*100) for t, count in tier_counts.items()},
                "token_estimates": prompt_system.get_token_estimates()
            },
            "performance": {
                "total_runtime": processing_time,
                "metrics_calculation_time": metrics_time,
                "sentences_per_minute": len(sentences)/processing_time*60 if processing_time > 0 else 0,
                "cache_hit_rate": cache_hits/len(results)*100 if cache and len(results) > 0 else 0,
                "api_calls": sum(1 for r in results if r.get("api_call", True))
            }
        }, f, indent=2)
    
    # Print final output
    print(f"\nResults saved to {output_dir}")
    print(f"Main results: {output_file}")
    print(f"Optimization metrics: {output_dir}/optimization_metrics.json")

if __name__ == "__main__":
    main()