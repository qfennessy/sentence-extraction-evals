#!/usr/bin/env python
"""
Phase 1 Optimizations Demo Script.

This script demonstrates how to use the Phase 1 optimizations implemented in claude_extraction_eval.py:
1. Parallel API Request Processing
2. Response Caching System
3. Adaptive Rate Limiting

Usage:
python phase1_demo.py --prompt-file prompt_template.txt --eval-data mini-family-sentences.json
"""

import os
import time
import click
from pathlib import Path

# Import the enhanced evaluator from claude_extraction_eval.py
from claude_extraction_eval import (
    evaluate_extraction, load_data, calculate_metrics, save_results,
    create_output_directory, AnthropicAdapter, CLAUDE_MODELS
)

@click.command()
@click.option("--prompt-file", required=True, type=click.Path(exists=True), help="Path to the prompt template file")
@click.option("--eval-data", required=True, type=click.Path(exists=True), help="Path to the evaluation data JSON file")
@click.option("--api-key", help="API key (or set via ANTHROPIC_API_KEY env var)")
@click.option("--model", default="sonnet", help="Model to use (opus/sonnet/haiku)")
@click.option("--batch-size", type=int, default=10, help="Number of sentences to evaluate in each batch")
@click.option("--max-sentences", type=int, default=None, help="Maximum number of sentences to evaluate")
@click.option("--parallel/--no-parallel", default=True, help="Process sentences in parallel")
@click.option("--max-workers", type=int, default=5, help="Maximum number of parallel workers")
def main(prompt_file, eval_data, api_key, model, batch_size, max_sentences, parallel, max_workers):
    """
    Demonstrate Phase 1 optimizations for extraction evaluation.
    """
    # Start timing
    start_time = time.time()
    
    # Get API key
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise click.UsageError("Anthropic API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable")
    
    # Convert model name to ID
    if model in CLAUDE_MODELS:
        model_id = CLAUDE_MODELS[model]
    else:
        model_id = model  # Assume it's already the full model ID
    
    # Create client
    client = AnthropicAdapter(api_key)
    
    # Load data
    prompt_template, sentences = load_data(prompt_file, eval_data, max_sentences)
    
    # Create output directory
    output_dir = create_output_directory(f"{model}-phase1", prompt_file, "phase1-demo")
    
    # Display configuration
    print(f"Phase 1 Optimizations Demo")
    print(f"=========================")
    print(f"Model: {model} ({model_id})")
    print(f"Prompt file: {prompt_file}")
    print(f"Eval data: {eval_data}")
    print(f"Sentences: {len(sentences)}")
    print(f"Optimizations:")
    print(f"  - Parallel processing: {'Enabled' if parallel else 'Disabled'}")
    if parallel:
        print(f"    - Max workers: {max_workers}")
    print(f"  - Response caching: Enabled")
    print(f"  - Adaptive rate limiting: Enabled")
    print(f"Output directory: {output_dir}")
    print(f"=========================")
    
    # First run - will use API calls and populate cache
    print("\nFirst run (API calls, populating cache)...")
    first_start = time.time()
    
    # Run evaluation with optimizations
    first_results = evaluate_extraction(
        client, 
        prompt_template, 
        sentences, 
        model_id, 
        batch_size,
        temperature=0.0,
        parallel=parallel,
        max_workers=max_workers
    )
    
    first_end = time.time()
    first_duration = first_end - first_start
    
    # Calculate and display metrics
    first_metrics = calculate_metrics(first_results)
    
    print(f"First run completed in {first_duration:.2f} seconds")
    print(f"Processing rate: {len(sentences)/first_duration*60:.2f} sentences per minute")
    print(f"Accuracy: {first_metrics.get('overall_value_accuracy', 0)*100:.2f}%")
    
    # Count cache hits in first run (should be 0)
    first_cache_hits = sum(1 for r in first_results if r.get("cached", False))
    print(f"Cache hits: {first_cache_hits}/{len(sentences)}")
    
    # Sleep briefly
    time.sleep(1)
    
    # Second run - should use cache for most responses
    print("\nSecond run (using cache)...")
    second_start = time.time()
    
    # Run evaluation again with same settings
    second_results = evaluate_extraction(
        client, 
        prompt_template, 
        sentences, 
        model_id, 
        batch_size,
        temperature=0.0,
        parallel=parallel,
        max_workers=max_workers
    )
    
    second_end = time.time()
    second_duration = second_end - second_start
    
    # Calculate and display metrics
    second_metrics = calculate_metrics(second_results)
    
    print(f"Second run completed in {second_duration:.2f} seconds")
    print(f"Processing rate: {len(sentences)/second_duration*60:.2f} sentences per minute")
    print(f"Accuracy: {second_metrics.get('overall_value_accuracy', 0)*100:.2f}%")
    
    # Count cache hits in second run (should be high)
    second_cache_hits = sum(1 for r in second_results if r.get("cached", False))
    print(f"Cache hits: {second_cache_hits}/{len(sentences)}")
    
    # Calculate speedup
    speedup = first_duration / second_duration if second_duration > 0 else float('inf')
    print(f"\nSpeedup from caching: {speedup:.2f}x")
    
    # Save results
    save_results(second_results, second_metrics, output_dir, model, eval_data, second_duration, prompt_file, "phase1-demo")
    
    # Save a summary report
    with open(output_dir / "phase1_summary.txt", 'w') as f:
        f.write("Phase 1 Optimization Results\n")
        f.write("==========================\n\n")
        f.write(f"Configuration:\n")
        f.write(f"- Model: {model} ({model_id})\n")
        f.write(f"- Sentences: {len(sentences)}\n")
        f.write(f"- Parallel Processing: {'Enabled' if parallel else 'Disabled'}")
        if parallel:
            f.write(f" (max_workers={max_workers})\n")
        else:
            f.write("\n")
        f.write(f"- Response Caching: Enabled\n")
        f.write(f"- Adaptive Rate Limiting: Enabled\n\n")
        
        f.write(f"Performance Results:\n")
        f.write(f"- First Run (API calls): {first_duration:.2f} seconds ({len(sentences)/first_duration*60:.2f} sentences/minute)\n")
        f.write(f"- Second Run (cached): {second_duration:.2f} seconds ({len(sentences)/second_duration*60:.2f} sentences/minute)\n")
        f.write(f"- Speedup: {speedup:.2f}x\n\n")
        
        f.write(f"Cache Statistics:\n")
        f.write(f"- First Run Cache Hits: {first_cache_hits}/{len(sentences)} ({first_cache_hits/len(sentences)*100:.2f}%)\n")
        f.write(f"- Second Run Cache Hits: {second_cache_hits}/{len(sentences)} ({second_cache_hits/len(sentences)*100:.2f}%)\n")
    
    print(f"\nResults saved to {output_dir}")
    print(f"Phase 1 summary: {output_dir}/phase1_summary.txt")

if __name__ == "__main__":
    main()