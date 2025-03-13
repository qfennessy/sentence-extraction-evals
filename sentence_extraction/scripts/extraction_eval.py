#!/usr/bin/env python
"""
Command-line interface for sentence extraction evaluation.

This script provides a CLI for evaluating models on sentence extraction tasks.
"""

import os
import time
import click
from pathlib import Path

# Import from the package
from sentence_extraction import (
    evaluate_extraction, 
    load_data, 
    calculate_metrics,
    save_results,
    create_output_directory,
    AnthropicAdapter, 
    OpenAIAdapter,
    GeminiAdapter,
    DeepSeekAdapter,
    CLAUDE_MODELS,
    OPENAI_MODELS,
    GEMINI_MODELS,
    DEEPSEEK_MODELS
)

@click.command()
@click.option("--prompt-file", required=True, type=click.Path(exists=True), help="Path to the prompt template file")
@click.option("--eval-data", required=True, type=click.Path(exists=True), help="Path to the evaluation data JSON file")
@click.option("--api-key", help="API key (or set via ANTHROPIC_API_KEY or OPENAI_API_KEY env var)")
@click.option("--model", 
    type=click.Choice(["opus", "sonnet", "haiku", "gpt4o", "gpt4", "gpt35", 
                      "pro", "pro-vision", "ultra", "coder", "chat"], 
                      case_sensitive=False), 
    default="sonnet", 
    help="Model to use (opus/sonnet/haiku for Claude, gpt4o/gpt4/gpt35 for OpenAI, pro/pro-vision/ultra for Gemini, coder/chat for DeepSeek)")
@click.option("--batch-size", type=int, default=5, help="Number of sentences to evaluate in each batch")
@click.option("--max-sentences", type=int, default=None, help="Maximum number of sentences to evaluate")
@click.option("--temperature", type=float, default=0.0, help="Temperature for model responses (0.0-1.0)")
@click.option("--parallel/--no-parallel", default=True, help="Process sentences in parallel (default: True)")
@click.option("--max-workers", type=int, default=5, help="Maximum number of parallel workers when parallel=True (default: 5)")
def main(prompt_file, eval_data, api_key, model, batch_size, max_sentences, temperature, parallel, max_workers):
    """Evaluate a model's ability to extract structured information from sentences."""
    # Start timing the evaluation
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
    output_dir = create_output_directory(model, prompt_file, model_type)
    
    click.echo(f"Loaded {len(sentences)} sentences for evaluation")
    click.echo(f"Using model: {model} ({model_id}) from {model_type.upper()}")
    click.echo(f"Temperature: {temperature}")
    click.echo(f"Parallelization: {'Enabled' if parallel else 'Disabled'}")
    if parallel:
        click.echo(f"Max workers: {max_workers}")
    click.echo(f"Output directory: {output_dir}")
    
    # Evaluate extraction
    results = evaluate_extraction(
        client, 
        prompt_template, 
        sentences, 
        model_id, 
        batch_size,
        temperature,
        parallel,
        max_workers
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Calculate total runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    
    # Print summary statistics
    click.echo("\nEvaluation Summary:")
    click.echo(f"Total sentences: {metrics['total_sentences']}")
    click.echo(f"Successful extractions: {metrics['successful_extractions']} ({metrics['successful_extractions']/metrics['total_sentences']*100:.2f}%)")
    click.echo(f"Failed extractions: {metrics['failed_extractions']} ({metrics['failed_extractions']/metrics['total_sentences']*100:.2f}%)")
    
    # Print performance statistics
    import datetime
    runtime_formatted = str(datetime.timedelta(seconds=int(runtime_seconds)))
    click.echo(f"\nPerformance Statistics:")
    click.echo(f"  Total runtime: {runtime_formatted}")
    click.echo(f"  Average time per sentence: {runtime_seconds/metrics['total_sentences']:.2f} seconds")
    processing_rate = metrics['total_sentences'] / runtime_seconds * 60 if runtime_seconds > 0 else 0
    click.echo(f"  Processing rate: {processing_rate:.2f} sentences per minute")
    
    # Save results
    output_file = save_results(results, metrics, output_dir, model, eval_data, runtime_seconds, prompt_file, model_type)
    
    click.echo(f"\nDetailed results saved to {output_dir}")

if __name__ == "__main__":
    main()