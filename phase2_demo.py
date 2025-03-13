#!/usr/bin/env python
"""
Phase 2 Optimizations Demo Script.

This script demonstrates how to use the Phase 2 optimizations implemented in 
the sentence_extraction package, including:
1. Tiered Prompt System
2. Optimized JSON Parsing
3. Metrics Calculation Optimization

Usage:
python phase2_demo.py --prompt-file prompt_template.txt --eval-data mini-family-sentences.json --tier auto
"""

import os
import time
import json
import click
from pathlib import Path
from typing import Dict, List, Any

# Import components from the package
from sentence_extraction import (
    # Core functionality
    AnthropicAdapter, 
    load_data,
    format_prompt,
    create_output_directory,
    save_results,
    CLAUDE_MODELS,
    
    # Phase 1 optimizations
    ResponseCache,
    ModelRateLimiter,
    process_data_parallel,
    
    # Phase 2 optimizations
    extract_json_efficiently,
    TieredPromptSystem,
    calculate_metrics_optimized
)

def process_with_tiered_prompt(
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
    Process a single item with tiered prompt system.
    
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
@click.option("--api-key", help="API key (or set via ANTHROPIC_API_KEY env var)")
@click.option("--model", default="sonnet", help="Model to use (opus/sonnet/haiku)")
@click.option("--tier", type=click.Choice(["fast", "standard", "comprehensive", "auto"]), default="auto", help="Prompt tier to use")
@click.option("--max-sentences", type=int, default=5, help="Maximum number of sentences to evaluate")
@click.option("--parallel/--no-parallel", default=True, help="Enable parallel processing")
@click.option("--max-workers", type=int, default=5, help="Maximum number of parallel workers")
@click.option("--detailed-metrics/--fast-metrics", default=True, help="Use detailed metrics calculation")
def main(prompt_file, eval_data, api_key, model, tier, max_sentences, parallel, max_workers, detailed_metrics):
    """
    Demonstrate Phase 2 optimizations for extraction evaluation.
    """
    # Start timing
    start_time = time.time()
    
    # Get API key
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise click.UsageError("Anthropic API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable")
    
    # Setup model
    if model in CLAUDE_MODELS:
        model_id = CLAUDE_MODELS[model]
    else:
        model_id = model  # Assume it's already the full model ID
    
    # Create client
    client = AnthropicAdapter(api_key)
    
    # Load data
    prompt_template, sentences = load_data(prompt_file, eval_data, max_sentences)
    
    # Create output directory
    output_dir = create_output_directory(f"{model}-phase2", prompt_file, "phase2-demo")
    
    # Initialize tiered prompt system
    prompt_system = TieredPromptSystem(prompt_file)
    
    # Initialize cache and rate limiter
    cache = ResponseCache(cache_dir="./cache")
    rate_limiter = ModelRateLimiter().get_limiter(model_id)
    
    # Display configuration
    print(f"Phase 2 Optimizations Demo")
    print(f"=========================")
    print(f"Model: {model} ({model_id})")
    print(f"Prompt file: {prompt_file}")
    print(f"Eval data: {eval_data}")
    print(f"Sentences: {len(sentences)}")
    print(f"Tier: {tier}")
    print(f"Optimizations:")
    print(f"  1. Tiered Prompt System: Enabled")
    print(f"  2. Optimized JSON Parsing: Enabled")
    print(f"  3. Metrics Calculation Optimization: {'Detailed' if detailed_metrics else 'Fast'}")
    print(f"  - Parallelization: {'Enabled' if parallel else 'Disabled'}")
    if parallel:
        print(f"    - Max workers: {max_workers}")
    print(f"  - Response caching: Enabled")
    print(f"  - Adaptive rate limiting: Enabled")
    print(f"Output directory: {output_dir}")
    print(f"=========================")
    
    # Print token estimates for each tier
    token_estimates = prompt_system.get_token_estimates()
    print("\nToken Estimates by Tier:")
    for t, tokens in token_estimates.items():
        print(f"  {t.capitalize()}: ~{tokens} tokens")
    
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
        # Parallel processing
        print(f"\nProcessing {len(sentences)} sentences in parallel with {max_workers} workers...")
        results = process_data_parallel(
            process_items,
            lambda item: process_with_tiered_prompt(
                client, prompt_system, item, model_id, tier, 
                0.0, rate_limiter, cache
            ),
            max_workers=max_workers
        )
    else:
        # Sequential processing
        print(f"\nProcessing {len(sentences)} sentences sequentially...")
        results = []
        for item in process_items:
            result = process_with_tiered_prompt(
                client, prompt_system, item, model_id, tier, 
                0.0, rate_limiter, cache
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
    print(f"\nProcessing Statistics:")
    print(f"  Total processing time: {processing_time:.2f} seconds")
    print(f"  Average time per sentence: {processing_time/len(sentences):.2f} seconds")
    print(f"  Processing rate: {len(sentences)/processing_time*60:.2f} sentences/minute")
    print(f"  Metrics calculation time: {metrics_time:.4f} seconds")
    
    # Print cache statistics
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
    
    # Save detailed results
    with open(output_dir / "phase2_results.json", 'w') as f:
        # Filter out full_response to reduce file size
        cleaned_results = []
        for result in results:
            result_copy = result.copy()
            if 'full_response' in result_copy:
                del result_copy['full_response']
            cleaned_results.append(result_copy)
            
        json.dump({
            "configuration": {
                "model": model,
                "model_id": model_id,
                "tier": tier,
                "parallel": parallel,
                "max_workers": max_workers,
                "detailed_metrics": detailed_metrics
            },
            "results": cleaned_results,
            "metrics": metrics,
            "tier_statistics": {
                "counts": tier_counts,
                "percentages": {t: (count/len(results)*100) for t, count in tier_counts.items()},
                "token_estimates": token_estimates
            },
            "performance": {
                "processing_time": processing_time,
                "metrics_time": metrics_time,
                "sentences_per_minute": len(sentences)/processing_time*60,
                "cache_hit_rate": cache_hits/len(results)*100 if len(results) > 0 else 0
            }
        }, f, indent=2)
    
    # Save a summary report
    with open(output_dir / "phase2_summary.md", 'w') as f:
        f.write("# Phase 2 Optimization Demo Results\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Model: {model} ({model_id})\n")
        f.write(f"- Prompt tier: {tier}\n")
        f.write(f"- Sentences processed: {len(sentences)}\n\n")
        
        f.write(f"## Tier Usage\n\n")
        f.write(f"| Tier | Count | Percentage |\n")
        f.write(f"|------|-------|------------|\n")
        for t, count in tier_counts.items():
            percentage = (count / len(results)) * 100
            f.write(f"| {t.capitalize()} | {count} | {percentage:.1f}% |\n")
        
        f.write(f"\n## Performance Metrics\n\n")
        f.write(f"- Total processing time: {processing_time:.2f} seconds\n")
        f.write(f"- Average time per sentence: {processing_time/len(sentences):.2f} seconds\n")
        f.write(f"- Processing rate: {len(sentences)/processing_time*60:.2f} sentences/minute\n")
        f.write(f"- Cache hit rate: {cache_hits/len(results)*100:.1f}%\n")
        f.write(f"- Metrics calculation time: {metrics_time:.4f} seconds\n\n")
        
        f.write(f"## Accuracy Metrics\n\n")
        f.write(f"- Overall accuracy: {metrics.get('overall_value_accuracy', 0)*100:.2f}%\n")
        f.write(f"- Successful extractions: {metrics['successful_extractions']}/{metrics['total_sentences']}\n")
        f.write(f"- Average score: {metrics['scores']['avg_score']*100:.2f}%\n")
        
        # Add sentence-level details
        f.write(f"\n## Sentence-Level Details\n\n")
        for i, result in enumerate(results):
            f.write(f"### Sentence {i+1}\n\n")
            f.write(f"```\n{result['sentence']}\n```\n\n")
            f.write(f"- Selected tier: {result.get('selected_tier', 'unknown')}\n")
            f.write(f"- Processing time: {result.get('processing_time', 0):.3f} seconds\n")
            f.write(f"- Cached: {'Yes' if result.get('cached', False) else 'No'}\n")
            
            # Add complexity analysis if available
            complexity = result.get('complexity_analysis')
            if complexity:
                f.write(f"- Complexity: {complexity.get('complexity', 'unknown')}\n")
                if complexity.get('high_complexity_matches'):
                    f.write(f"- High complexity patterns: {', '.join(complexity.get('high_complexity_matches'))}\n")
                if complexity.get('medium_complexity_matches'):
                    f.write(f"- Medium complexity patterns: {', '.join(complexity.get('medium_complexity_matches'))}\n")
            
            f.write("\n")
    
    print(f"\nResults saved to {output_dir}")
    print(f"Detailed results: {output_dir}/phase2_results.json")
    print(f"Summary report: {output_dir}/phase2_summary.md")

if __name__ == "__main__":
    main()