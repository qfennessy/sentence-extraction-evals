# Sentence Extraction Framework Examples

This document provides practical examples for using the Family Information Extraction Evaluation Framework.

## Basic Examples

### Evaluating a Single Model

Evaluate how well a model extracts family information:

```python
from sentence_extraction import AnthropicAdapter, evaluate_extraction, load_data, calculate_metrics
import os
import json
from pprint import pprint

# Initialize model client with API key from environment
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)

# Load prompt and data
with open("prompt_template.txt", "r") as f:
    prompt_template = f.read().strip()

with open("example_evaluation_data.json", "r") as f:
    data = json.load(f)
    sentences = data.get("sentences", [])

# Run evaluation
model = "claude-3-7-sonnet-20250219"
results = evaluate_extraction(
    client=client,
    prompt_template=prompt_template,
    sentences=sentences,
    model=model,
    batch_size=5,
    temperature=0.0,
    parallel=True,
    max_workers=5
)

# Calculate metrics
metrics = calculate_metrics(results)

# Print results
print(f"Overall value accuracy: {metrics['overall_value_accuracy'] * 100:.2f}%")
print(f"Field precision: {metrics['field_precision'] * 100:.2f}%")
print(f"Field recall: {metrics['field_recall'] * 100:.2f}%")
print(f"Field F1 score: {metrics['field_f1'] * 100:.2f}%")

# Print first result in detail
print("\nSample extraction:")
print(f"Sentence: {results[0]['sentence']}")
print("Expected:")
pprint(results[0]['expected'])
print("Extracted:")
pprint(results[0]['extracted'])
```

### Using the Command-Line Interface

```bash
# Basic evaluation
python claude_extraction_eval.py \
  --model sonnet \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --max-sentences 10 \
  --parallel \
  --max-workers 5
```

### Working with Different Model Providers

```python
from sentence_extraction import (
    AnthropicAdapter, 
    OpenAIAdapter, 
    GeminiAdapter, 
    DeepSeekAdapter,
    evaluate_extraction, 
    load_data
)
import os

# Load data
prompt_template, sentences = load_data("prompt_template.txt", "example_evaluation_data.json", max_sentences=5)

# Choose a subset for demonstration
test_sentences = sentences[:5]

# Initialize clients
anthropic_client = AnthropicAdapter(os.environ.get("ANTHROPIC_API_KEY"))
openai_client = OpenAIAdapter(os.environ.get("OPENAI_API_KEY"))
gemini_client = GeminiAdapter(os.environ.get("GOOGLE_API_KEY"))
deepseek_client = DeepSeekAdapter(os.environ.get("DEEPSEEK_API_KEY"))

# Define models
models = {
    "claude": ("sonnet", anthropic_client),
    "openai": ("gpt4o", openai_client),
    "gemini": ("pro", gemini_client),
    "deepseek": ("chat", deepseek_client)
}

# Run evaluations
results = {}
for provider, (model, client) in models.items():
    print(f"Evaluating {provider} {model}...")
    results[provider] = evaluate_extraction(
        client=client,
        prompt_template=prompt_template,
        sentences=test_sentences,
        model=model,
        parallel=True,
        max_workers=5
    )

# Compare results
for provider, provider_results in results.items():
    accuracy = sum(r.get("metrics", {}).get("value_accuracy", 0) for r in provider_results) / len(provider_results)
    print(f"{provider} accuracy: {accuracy * 100:.2f}%")
```

## Optimization Examples

### Using All Optimizations

```python
from sentence_extraction.optimizations import (
    ResponseCache, 
    extract_json_efficiently,
    TieredPromptSystem,
    AdaptiveRateLimiter,
    calculate_metrics_optimized
)
from sentence_extraction import AnthropicAdapter, format_prompt
import os
import time
import json

# Initialize components
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)
cache = ResponseCache("./cache")
rate_limiter = AdaptiveRateLimiter(requests_per_minute=3.0)
prompt_system = TieredPromptSystem("prompt_template.txt")

# Load sentences
with open("example_evaluation_data.json", "r") as f:
    data = json.load(f)
    sentences = data.get("sentences", [])[:10]  # Process first 10

# Process with optimizations
model = "claude-3-7-sonnet-20250219"
results = []

for item in sentences:
    sentence = item["sentence"]
    expected = item.get("extracted_information", {})
    
    # Select appropriate tier
    tier = prompt_system.get_recommended_tier(sentence)
    prompt_template = prompt_system.get_prompt(tier)
    
    # Format prompt
    prompt = format_prompt(prompt_template, sentence)
    
    # Check cache
    cached_response = cache.get(prompt, sentence, model)
    
    # Apply rate limiting and call API if needed
    if cached_response:
        response = cached_response
        from_cache = True
    else:
        rate_limiter.wait()
        response = client(prompt, model)
        cache.set(prompt, sentence, model, response)
        from_cache = False
    
    # Extract JSON efficiently
    extracted = extract_json_efficiently(response)
    
    # Store result
    results.append({
        "sentence": sentence,
        "expected": expected,
        "extracted": extracted,
        "from_cache": from_cache,
        "selected_tier": tier
    })

# Calculate optimized metrics
metrics = calculate_metrics_optimized(results)

# Print results
print(f"Overall accuracy: {metrics['overall_value_accuracy'] * 100:.2f}%")
print(f"Tier distribution: {[r['selected_tier'] for r in results].count('fast')} fast, "
      f"{[r['selected_tier'] for r in results].count('standard')} standard, "
      f"{[r['selected_tier'] for r in results].count('comprehensive')} comprehensive")
print(f"Cache hits: {sum(1 for r in results if r['from_cache'])}/{len(results)}")
```

### Parallel Processing Example

```python
from sentence_extraction.optimizations import ParallelProcessor
from sentence_extraction import AnthropicAdapter, format_prompt, extract_json_from_response
import os
import time

# Initialize components
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)
processor = ParallelProcessor(max_workers=5)

# Sample data
with open("prompt_template.txt", "r") as f:
    prompt_template = f.read().strip()

import json
with open("example_evaluation_data.json", "r") as f:
    data = json.load(f)
    sentences = data.get("sentences", [])[:20]  # Process first 20

# Define processing function
def process_sentence(item):
    sentence = item["sentence"]
    prompt = format_prompt(prompt_template, sentence)
    response = client(prompt, "claude-3-7-sonnet-20250219")
    extracted = extract_json_from_response(response)
    return {
        "sentence": sentence,
        "extracted": extracted
    }

# Process in parallel
start_time = time.time()
parallel_results = processor.process_batch(sentences, process_sentence, "Processing sentences")
parallel_time = time.time() - start_time

# Process sequentially for comparison
start_time = time.time()
sequential_results = [process_sentence(item) for item in sentences]
sequential_time = time.time() - start_time

# Compare performance
print(f"Parallel processing time: {parallel_time:.2f} seconds")
print(f"Sequential processing time: {sequential_time:.2f} seconds")
print(f"Speedup: {sequential_time / parallel_time:.2f}x")
```

### Tiered Prompt System Example

```python
from sentence_extraction.optimizations import TieredPromptSystem
from sentence_extraction import AnthropicAdapter
import os
import json
from pprint import pprint

# Initialize components
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)
prompt_system = TieredPromptSystem("prompt_template.txt")

# Load sample sentences
with open("example_evaluation_data.json", "r") as f:
    data = json.load(f)
    sentences = [item["sentence"] for item in data.get("sentences", [])][:3]

# Test each tier
results = {}
model = "claude-3-7-sonnet-20250219"

for tier in ["fast", "standard", "comprehensive"]:
    prompt_template = prompt_system.get_prompt(tier)
    
    tier_results = []
    for sentence in sentences:
        # Format prompt with sentence
        prompt = prompt_template.replace("{SENTENCE}", sentence)
        
        # Call model
        response = client(prompt, model)
        
        # Store result
        tier_results.append({
            "sentence": sentence,
            "response": response
        })
    
    results[tier] = tier_results

# Compare response lengths
for tier, tier_results in results.items():
    avg_length = sum(len(r["response"]) for r in tier_results) / len(tier_results)
    print(f"{tier} tier average response length: {avg_length:.1f} characters")

# Analyze a specific sentence
sample_sentence = sentences[0]
complexity = prompt_system.analyze_sentence_complexity(sample_sentence)
recommended_tier = prompt_system.get_recommended_tier(sample_sentence)

print(f"\nSample sentence: {sample_sentence}")
print(f"Complexity metrics: {complexity}")
print(f"Recommended tier: {recommended_tier}")
```

## Advanced Examples

### Cross-Validation

```python
from sentence_extraction import cross_validation, AnthropicAdapter
import os
import json
from pprint import pprint

# Initialize client
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)

# Define parameters
datasets = ["very-simple-family-sentences-evals.json", "family-sentences-evals.json"]
models = ["sonnet", "haiku"]
prompt_variations = ["prompt_template.txt", "prompt_hierarchical.txt"]
folds = 3
max_sentences = 15

# Run cross-validation
results = cross_validation(
    client=client,
    datasets=datasets,
    models=models,
    prompt_variations=prompt_variations,
    folds=folds,
    max_sentences=max_sentences,
    output_dir="cross_validation_results"
)

# Print summary
print("Cross-Validation Results Summary:")
for dataset in datasets:
    print(f"\nDataset: {dataset}")
    for model in models:
        print(f"  Model: {model}")
        for prompt in prompt_variations:
            accuracy = results[dataset][model][prompt]["overall_value_accuracy"]
            print(f"    Prompt: {prompt} - Accuracy: {accuracy * 100:.2f}%")

# Find best configuration
best_accuracy = 0
best_config = None

for dataset in datasets:
    for model in models:
        for prompt in prompt_variations:
            accuracy = results[dataset][model][prompt]["overall_value_accuracy"]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = (dataset, model, prompt)

print(f"\nBest configuration: {best_config} with {best_accuracy * 100:.2f}% accuracy")
```

### Prompt Refinement

```python
from sentence_extraction import (
    AnthropicAdapter, 
    refine_prompt, 
    load_data, 
    split_dataset, 
    evaluate_extraction
)
import os
import tempfile
import json
from pprint import pprint

# Initialize client
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)

# Split dataset
dataset_path = "family-sentences-evals.json"
with tempfile.TemporaryDirectory() as temp_dir:
    splits = split_dataset(
        dataset_path=dataset_path,
        output_dir=temp_dir,
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2
    )
    
    # Load splits
    with open(splits["train"], "r") as f:
        train_data = json.load(f)["sentences"]
    
    with open(splits["validation"], "r") as f:
        validation_data = json.load(f)["sentences"]
    
    with open(splits["test"], "r") as f:
        test_data = json.load(f)["sentences"]
    
    # Load initial prompt
    with open("prompt_template.txt", "r") as f:
        base_prompt = f.read().strip()
    
    # Refine prompt
    refined_prompt, refinement_metrics = refine_prompt(
        base_prompt=base_prompt,
        train_data=train_data,
        validation_data=validation_data,
        model="sonnet",
        max_iterations=3,
        client=client
    )
    
    # Save refined prompt
    refined_prompt_path = os.path.join(temp_dir, "refined_prompt.txt")
    with open(refined_prompt_path, "w") as f:
        f.write(refined_prompt)
    
    # Evaluate original vs refined prompt on test set
    print("Evaluating original prompt...")
    original_results = evaluate_extraction(
        client=client,
        prompt_template=base_prompt,
        sentences=test_data,
        model="claude-3-7-sonnet-20250219",
        parallel=True
    )
    
    print("Evaluating refined prompt...")
    refined_results = evaluate_extraction(
        client=client,
        prompt_template=refined_prompt,
        sentences=test_data,
        model="claude-3-7-sonnet-20250219",
        parallel=True
    )
    
    # Calculate overall accuracy
    original_accuracy = sum(r.get("metrics", {}).get("value_accuracy", 0) for r in original_results) / len(original_results)
    refined_accuracy = sum(r.get("metrics", {}).get("value_accuracy", 0) for r in refined_results) / len(refined_results)
    
    # Print results
    print(f"Original prompt accuracy: {original_accuracy * 100:.2f}%")
    print(f"Refined prompt accuracy: {refined_accuracy * 100:.2f}%")
    print(f"Improvement: {(refined_accuracy - original_accuracy) * 100:.2f} percentage points")
    
    # Show refinement history
    print("\nPrompt refinement metrics:")
    for i, metrics in enumerate(refinement_metrics.get("iterations", [])):
        print(f"Iteration {i+1}: {metrics.get('accuracy', 0) * 100:.2f}% accuracy")
```

### Response Caching Example

```python
from sentence_extraction.optimizations import ResponseCache
from sentence_extraction import AnthropicAdapter, format_prompt
import os
import time
import json

# Initialize components
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)
cache = ResponseCache("./cache", expiration_hours=24)

# Clear cache to start fresh
cache.clear_all()

# Load data
with open("prompt_template.txt", "r") as f:
    prompt_template = f.read().strip()

with open("example_evaluation_data.json", "r") as f:
    data = json.load(f)
    sentences = data.get("sentences", [])[:5]  # Process first 5

model = "claude-3-7-sonnet-20250219"

# First run (no cache)
print("First run (no cache)...")
start_time = time.time()
for item in sentences:
    sentence = item["sentence"]
    prompt = format_prompt(prompt_template, sentence)
    
    # Check cache (should be miss)
    cached = cache.get(prompt, sentence, model)
    if cached:
        print(f"Cache hit for: {sentence}")
        response = cached
    else:
        print(f"Cache miss for: {sentence}")
        response = client(prompt, model)
        cache.set(prompt, sentence, model, response)

first_run_time = time.time() - start_time
print(f"First run completed in {first_run_time:.2f} seconds")

# Second run (with cache)
print("\nSecond run (with cache)...")
start_time = time.time()
for item in sentences:
    sentence = item["sentence"]
    prompt = format_prompt(prompt_template, sentence)
    
    # Check cache (should be hit)
    cached = cache.get(prompt, sentence, model)
    if cached:
        print(f"Cache hit for: {sentence}")
        response = cached
    else:
        print(f"Cache miss for: {sentence}")
        response = client(prompt, model)
        cache.set(prompt, sentence, model, response)

second_run_time = time.time() - start_time
print(f"Second run completed in {second_run_time:.2f} seconds")

# Show statistics
cache_stats = cache.get_stats()
print("\nCache statistics:")
print(f"Total entries: {cache_stats.get('total_entries', 0)}")
print(f"Total size: {cache_stats.get('total_size_mb', 0):.2f} MB")
print(f"Speedup from caching: {first_run_time / second_run_time:.2f}x")
```

## Specialized Examples

### JSON Extraction Comparison

```python
from sentence_extraction.optimizations import extract_json_efficiently
from sentence_extraction import extract_json_from_response
import json
import time

# Sample responses with different JSON formats
samples = [
    '''Here's the extracted information:
```json
{
  "name": "David",
  "relationship": "Father",
  "occupation": "mechanic",
  "location": "Chicago",
  "work_duration": "35 years"
}
```''',
    '''The extracted information is:
{
  "name": "Emily",
  "relationship": "Sister",
  "education": "Harvard",
  "degree": "Biology",
  "graduation_year": "2018"
}''',
    '''I've analyzed the sentence and here's what I found:
{
  "family_members": [
    {
      "name": "David",
      "relationship": "Father"
    },
    {
      "name": "Sarah",
      "relationship": "Mother"
    }
  ]
}'''
]

# Compare extraction methods
for i, sample in enumerate(samples):
    print(f"\nSample {i+1}:")
    
    # Standard extraction
    start_time = time.time()
    try:
        standard_result = extract_json_from_response(sample)
        standard_time = time.time() - start_time
        print(f"Standard extraction: Success in {standard_time:.6f} seconds")
    except Exception as e:
        standard_time = time.time() - start_time
        print(f"Standard extraction: Failed in {standard_time:.6f} seconds - {str(e)}")
    
    # Optimized extraction
    start_time = time.time()
    try:
        optimized_result = extract_json_efficiently(sample)
        optimized_time = time.time() - start_time
        print(f"Optimized extraction: Success in {optimized_time:.6f} seconds")
    except Exception as e:
        optimized_time = time.time() - start_time
        print(f"Optimized extraction: Failed in {optimized_time:.6f} seconds - {str(e)}")
    
    if 'standard_result' in locals() and 'optimized_result' in locals():
        print("Results match:", standard_result == optimized_result)
```

### Complex Dataset Processing

```python
from sentence_extraction import (
    AnthropicAdapter, 
    evaluate_extraction, 
    calculate_metrics,
    compare_extraction_results
)
import os
import json
from pprint import pprint

# Initialize client
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)

# Load complex data and prompt
with open("complex-family-sentences-evals.json", "r") as f:
    complex_data = json.load(f)
    complex_sentences = complex_data.get("sentences", [])[:5]  # First 5 examples

with open("prompt_hierarchical.txt", "r") as f:
    complex_prompt_template = f.read().strip()

# Run evaluation on complex examples
model = "claude-3-7-sonnet-20250219"
complex_results = evaluate_extraction(
    client=client,
    prompt_template=complex_prompt_template,
    sentences=complex_sentences,
    model=model,
    parallel=True,
    max_workers=5
)

# Calculate metrics
metrics = calculate_metrics(complex_results)

# Print results
print(f"Complex dataset metrics:")
print(f"Overall value accuracy: {metrics['overall_value_accuracy'] * 100:.2f}%")
print(f"Field precision: {metrics['field_precision'] * 100:.2f}%")
print(f"Field recall: {metrics['field_recall'] * 100:.2f}%")

# Analyze a complex extraction example with multiple family members
example = complex_results[0]
print("\nComplex extraction example:")
print(f"Sentence: {example['sentence']}")

# Detailed comparison
comparison = compare_extraction_results(example['expected'], example['extracted'])
print(f"\nComparison metrics:")
print(f"Field matches: {comparison.get('field_matches', 0)}")
print(f"Field errors: {comparison.get('field_errors', 0)}")
print(f"Value accuracy: {comparison.get('value_accuracy', 0) * 100:.2f}%")

# Check for multiple family members
expected_members = example['expected'].get('family_members', [])
extracted_members = example['extracted'].get('family_members', [])

print(f"\nExpected family members: {len(expected_members)}")
print(f"Extracted family members: {len(extracted_members)}")
```

### Domain-Specific Extraction

```python
from sentence_extraction import AnthropicAdapter, evaluate_extraction, calculate_metrics
import os
import json

# Initialize client
api_key = os.environ.get("ANTHROPIC_API_KEY")
client = AnthropicAdapter(api_key)

# Load domain-specific data and prompts
with open("domain_specific_test.json", "r") as f:
    domain_data = json.load(f)
    domain_sentences = domain_data.get("sentences", [])

with open("test_domain_prompt.txt", "r") as f:
    domain_prompt = f.read().strip()

with open("prompt_template.txt", "r") as f:
    standard_prompt = f.read().strip()

# Run evaluations with both prompts
model = "claude-3-7-sonnet-20250219"

domain_results = evaluate_extraction(
    client=client,
    prompt_template=domain_prompt,
    sentences=domain_sentences,
    model=model,
    parallel=True
)

standard_results = evaluate_extraction(
    client=client,
    prompt_template=standard_prompt,
    sentences=domain_sentences,
    model=model,
    parallel=True
)

# Calculate metrics
domain_metrics = calculate_metrics(domain_results)
standard_metrics = calculate_metrics(standard_results)

# Compare results
print("Domain-Specific vs Standard Prompt Comparison:")
print(f"Domain prompt accuracy: {domain_metrics['overall_value_accuracy'] * 100:.2f}%")
print(f"Standard prompt accuracy: {standard_metrics['overall_value_accuracy'] * 100:.2f}%")
print(f"Improvement: {(domain_metrics['overall_value_accuracy'] - standard_metrics['overall_value_accuracy']) * 100:.2f} percentage points")

# Analyze field-specific improvements
print("\nField-specific improvements:")
domain_field_acc = domain_metrics.get("field_accuracy", {})
standard_field_acc = standard_metrics.get("field_accuracy", {})

for field in domain_field_acc:
    if field in standard_field_acc:
        domain_acc = domain_field_acc[field]
        standard_acc = standard_field_acc[field]
        improvement = domain_acc - standard_acc
        print(f"{field}: {improvement * 100:.2f}% improvement")
```

## Command-Line Examples

### Cross-Validation

```bash
# Run cross-validation across datasets and models
python cross_validation.py \
  --datasets family-sentences-evals.json complex-family-sentences-evals.json \
  --models sonnet gpt4o \
  --prompt-variations prompt_template.txt prompt_hierarchical.txt \
  --folds 5 \
  --max-sentences 20 \
  --output-dir cv_results
```

### Optimized Evaluation

```bash
# Run fully optimized evaluation with tiered prompts
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data family-sentences-evals.json \
  --model sonnet \
  --tier auto \
  --parallel \
  --max-workers 5 \
  --cache \
  --rate-limiting
```

### Dataset Splitting

```bash
# Split a dataset for training/validation/testing
python dataset_splitter.py \
  family-sentences-evals.json \
  --output-dir test_split \
  --train-ratio 0.7 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --random-seed 42
```

### Prompt Refinement

```bash
# Run automated prompt refinement
python run_prompt_refinement.py \
  --base-prompt prompt_template.txt \
  --dataset family-sentences-evals.json \
  --output-dir prompt_refinement_workflow \
  --model sonnet \
  --max-iterations 5
```

### Model Comparison

```bash
# Compare multiple models on the same dataset
python run_model_comparison.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --models sonnet opus haiku gpt4o gpt4
```