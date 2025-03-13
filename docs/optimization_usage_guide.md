# Optimization Usage Guide

This guide explains how to use the optimized extraction system with both Phase 1 and Phase 2 optimizations.

## Overview

The extraction optimization system provides several performance enhancements:

**Phase 1 Optimizations**:
- **Parallel Processing**: Process multiple requests concurrently
- **Response Caching**: Avoid redundant API calls
- **Adaptive Rate Limiting**: Optimize request rates

**Phase 2 Optimizations**:
- **Tiered Prompt System**: Different prompt tiers for performance/accuracy tradeoffs
- **Optimized JSON Parsing**: Faster and more robust extraction of JSON
- **Metrics Calculation Optimization**: Early termination and selective computation

## Installation

Install the package in development mode:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/example/sentence-extraction-evals.git
cd sentence-extraction-evals

# Install in development mode
pip install -e .
```

## Command-line Usage

### Using Fully Optimized Evaluation

The simplest way to use all optimizations is through the `fully_optimized_extraction_eval.py` script:

```bash
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --model sonnet \
  --tier auto \
  --parallel \
  --max-workers 8 \
  --batch-size 10
```

Options:
- `--prompt-file`: Path to the prompt template file
- `--eval-data`: Path to the evaluation data JSON file
- `--model`: Model to use (sonnet, opus, haiku, gpt4o, etc.)
- `--tier`: Prompt tier to use (fast, standard, comprehensive, auto)
- `--parallel/--no-parallel`: Enable/disable parallel processing
- `--max-workers`: Maximum number of parallel workers
- `--batch-size`: Batch size for processing
- `--detailed-metrics/--fast-metrics`: Use detailed or fast metrics calculation
- `--cache/--no-cache`: Enable/disable response caching
- `--rate-limiting/--no-rate-limiting`: Enable/disable adaptive rate limiting

### Phase 1 Optimizations Demo

To focus on Phase 1 optimizations:

```bash
python phase1_demo.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --model sonnet \
  --parallel \
  --max-workers 5
```

### Phase 2 Optimizations Demo

To focus on Phase 2 optimizations:

```bash
python phase2_demo.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --model sonnet \
  --tier auto
```

## Using as a Python Library

### Importing Components

```python
from sentence_extraction import (
    # Core functionality
    AnthropicAdapter, load_data, evaluate_extraction,
    
    # Phase 1 optimizations
    ResponseCache, ModelRateLimiter, process_data_parallel,
    
    # Phase 2 optimizations
    extract_json_efficiently, TieredPromptSystem, calculate_metrics_optimized
)
```

### Basic Usage with Phase 1 Optimizations

```python
# Initialize client
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)

# Load data
prompt_template, sentences = load_data("prompt_template.txt", "eval_data.json")

# Evaluate with Phase 1 optimizations
results = evaluate_extraction(
    client,
    prompt_template,
    sentences,
    "claude-3-7-sonnet-20250219",  # Model ID
    batch_size=10,                 # Processing batch size
    parallel=True,                 # Enable parallel processing
    max_workers=8                  # Number of parallel workers
)
```

### Using Phase 2 Tiered Prompt System

```python
# Initialize tiered prompt system
prompt_system = TieredPromptSystem("prompt_template.txt")

# Process a sentence with adaptive tier selection
# First determine the appropriate tier for the sentence
sentence = "My father John is a doctor."
selected_tier = prompt_system.get_recommended_tier(sentence)

# Get the prompt for the selected tier
prompt_template = prompt_system.get_prompt(selected_tier)

# Format the prompt and call the model
prompt = prompt_template.replace("{SENTENCE}", sentence)
response = client(prompt, model_id)

# Extract JSON using optimized parser
result = extract_json_efficiently(response)
```

### Using Optimized Metrics Calculation

```python
# Calculate metrics with full details
detailed_metrics = calculate_metrics_optimized(results, detailed_metrics=True)

# Calculate metrics with minimal computation for speed
fast_metrics = calculate_metrics_optimized(results, detailed_metrics=False)

# Access the metrics
accuracy = detailed_metrics.get("overall_value_accuracy", 0)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

### Custom Processing Pipeline

For a fully customized pipeline, implement a processing function:

```python
def process_with_all_optimizations(client, prompt_system, item, model_id, tier="auto", cache=None, rate_limiter=None):
    sentence = item["sentence"]
    
    # Select tier based on complexity
    selected_tier = prompt_system.get_recommended_tier(sentence) if tier == "auto" else tier
    prompt_template = prompt_system.get_prompt(selected_tier)
    
    # Format prompt
    prompt = prompt_template.replace("{SENTENCE}", sentence)
    
    # Apply rate limiting
    if rate_limiter:
        rate_limiter.wait()
    
    # Check cache
    cached_response = None
    if cache:
        cached_response = cache.get(prompt, sentence, model_id)
    
    # Call API or use cached response
    if cached_response:
        response_text = cached_response
    else:
        response_text = client(prompt, model_id)
        if cache:
            cache.set(prompt, sentence, model_id, response_text)
    
    # Extract JSON with optimized parser
    extracted = extract_json_efficiently(response_text)
    
    return {
        "sentence": sentence,
        "extracted": extracted,
        "selected_tier": selected_tier
    }
```

## Performance Optimization Tips

### Maximizing Speed

For the fastest possible processing:

1. **Use the fast prompt tier**: `--tier fast`
2. **Enable parallel processing**: `--parallel --max-workers 8`
3. **Use response caching**: `--cache`
4. **Use fast metrics calculation**: `--fast-metrics`

Example:
```bash
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data eval_data.json \
  --model sonnet \
  --tier fast \
  --parallel \
  --max-workers 8 \
  --cache \
  --fast-metrics
```

### Balancing Speed and Accuracy

For a good balance of speed and accuracy:

1. **Use the auto tier selection**: `--tier auto`
2. **Enable parallel processing with moderate workers**: `--parallel --max-workers 5`
3. **Use detailed metrics**: `--detailed-metrics`

Example:
```bash
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data eval_data.json \
  --model sonnet \
  --tier auto \
  --parallel \
  --max-workers 5 \
  --detailed-metrics
```

### Maximizing Accuracy

For highest possible accuracy:

1. **Use the comprehensive prompt tier**: `--tier comprehensive`
2. **Enable detailed metrics**: `--detailed-metrics`

Example:
```bash
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data eval_data.json \
  --model sonnet \
  --tier comprehensive \
  --detailed-metrics
```

## Monitoring Performance

After running an evaluation, review the generated files in the output directory:

- `extraction_results.json`: Complete evaluation results
- `optimization_metrics.json`: Performance metrics for optimizations
- `summary.csv`: Overall extraction statistics
- `performance_metrics.json`: Detailed performance data

## Troubleshooting

### Cache-Related Issues

If you encounter issues with cached responses:

```bash
# Clear the cache
python -c "from sentence_extraction.optimizations import ResponseCache; ResponseCache('./cache').clear_all()"
```

### Parallel Processing Issues

If parallel processing causes errors:

```bash
# Try with sequential processing
python fully_optimized_extraction_eval.py --no-parallel
```

### Memory Usage Concerns

For very large datasets, use batching to manage memory:

```bash
python fully_optimized_extraction_eval.py --batch-size 5 --max-sentences 1000
```

## Getting Help

For more information, see:
- `phase1_implementation_summary.md`: Details on Phase 1 optimizations
- `phase2_implementation_summary.md`: Details on Phase 2 optimizations
- `docs/extraction_speed_optimization.md`: Original optimization plan