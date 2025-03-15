# Sentence Extraction Framework User Guide

## Introduction

The Family Information Extraction Evaluation Framework evaluates how well language models (LLMs) extract structured family information from text. This user guide covers installation, basic usage, advanced features, and optimization recommendations.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip
- Git (optional)

### Installation Options

#### Option 1: Direct Installation

```bash
# Clone the repository (optional)
git clone https://github.com/example/sentence-extraction-evals.git
cd sentence-extraction-evals

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Package Installation (Recommended)

```bash
# Clone the repository (optional)
git clone https://github.com/example/sentence-extraction-evals.git
cd sentence-extraction-evals

# Install as a package in development mode
pip install -e .

# For development with additional tools
pip install -e ".[dev]"
```

## Getting Started

### Setting Up API Keys

Store your API keys as environment variables:

```bash
# For Anthropic Claude models
export ANTHROPIC_API_KEY=your_anthropic_api_key

# For OpenAI models (GPT-4, GPT-3.5)
export OPENAI_API_KEY=your_openai_api_key

# For Google Gemini models
export GOOGLE_API_KEY=your_google_api_key

# For DeepSeek models
export DEEPSEEK_API_KEY=your_deepseek_api_key
```

For persistent storage, add these to your `.bashrc` or `.zshrc` file.

### Running a Simple Evaluation

#### Using the Command-Line Interface

```bash
# If installed as a package
extraction-eval --model sonnet --prompt-file prompt_template.txt --eval-data example_evaluation_data.json

# Or using the script directly
python claude_extraction_eval.py --model sonnet --prompt-file prompt_template.txt --eval-data example_evaluation_data.json
```

#### Basic Options

- `--model`: Model to use (sonnet, opus, haiku, gpt4o, gpt4, gpt35, pro, ultra, coder, chat)
- `--prompt-file`: Path to the prompt template file
- `--eval-data`: Path to the evaluation data file
- `--max-sentences`: Maximum number of sentences to evaluate
- `--temperature`: Temperature for model responses (default: 0.0)
- `--parallel/--no-parallel`: Enable/disable parallel processing (default: enabled)
- `--max-workers`: Number of parallel workers (default: 5)

### Using as a Python Library

```python
from sentence_extraction import (
    evaluate_extraction, 
    AnthropicAdapter, 
    load_data, 
    calculate_metrics
)

# Initialize client (API key from environment variable)
client = AnthropicAdapter()

# Load data
prompt_template, sentences = load_data("prompt_template.txt", "example_evaluation_data.json")

# Run evaluation
results = evaluate_extraction(
    client=client,
    prompt_template=prompt_template,
    sentences=sentences,
    model="claude-3-7-sonnet-20250219",
    batch_size=10,
    temperature=0.0,
    parallel=True,
    max_workers=5
)

# Calculate and display metrics
metrics = calculate_metrics(results)
print(f"Overall value accuracy: {metrics['overall_value_accuracy'] * 100:.2f}%")
print(f"Field precision: {metrics['field_precision'] * 100:.2f}%")
print(f"Field recall: {metrics['field_recall'] * 100:.2f}%")
```

## Data Formats

### Evaluation Data Format

Create JSON files with this structure:

```json
{
  "sentences": [
    {
      "sentence": "My father David worked as a mechanic in Chicago for 35 years.",
      "extracted_information": {
        "name": "David",
        "relationship": "Father",
        "occupation": "mechanic",
        "location": "Chicago",
        "work_duration": "35 years"
      }
    },
    {
      "sentence": "My sister Emily graduated from Harvard with a degree in Biology in 2018.",
      "extracted_information": {
        "name": "Emily",
        "relationship": "Sister",
        "education": "Harvard",
        "degree": "Biology",
        "graduation_year": "2018"
      }
    }
  ]
}
```

For complex family structures with multiple family members in one sentence:

```json
{
  "sentences": [
    {
      "sentence": "My father David and mother Sarah both grew up in Chicago.",
      "extracted_information": {
        "family_members": [
          {
            "name": "David",
            "relationship": "Father",
            "grew_up_in": "Chicago"
          },
          {
            "name": "Sarah",
            "relationship": "Mother",
            "grew_up_in": "Chicago"
          }
        ]
      }
    }
  ]
}
```

### Prompt Template Format

Create a text file with your prompt template, including a `{SENTENCE}` placeholder:

```
Extract structured information about family members from the following sentence.

Sentence: {SENTENCE}

Output the information in valid JSON format with these fields:
- name: The person's name
- relationship: Their relationship to the speaker
- Include any additional information mentioned:
  - occupation
  - location
  - age
  - education
  - achievements

JSON response:
```

## Advanced Features

### Performance Optimizations

#### Using the Fully Optimized Mode

For maximum performance, use the fully optimized evaluation script:

```bash
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --model sonnet \
  --tier auto \
  --parallel \
  --max-workers 8 \
  --cache
```

Additional options:
- `--tier`: Prompt tier to use (fast, standard, comprehensive, auto)
- `--cache/--no-cache`: Enable/disable response caching
- `--rate-limiting/--no-rate-limiting`: Enable/disable adaptive rate limiting
- `--detailed-metrics/--fast-metrics`: Use detailed or fast metrics calculation

#### Optimization Configurations

For high-speed processing:
```bash
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --model sonnet \
  --tier fast \
  --parallel \
  --max-workers 8 \
  --cache \
  --fast-metrics
```

For balanced performance:
```bash
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --model sonnet \
  --tier auto \
  --parallel \
  --max-workers 5
```

For maximum accuracy:
```bash
python fully_optimized_extraction_eval.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --model sonnet \
  --tier comprehensive \
  --detailed-metrics
```

### Cross-Validation Framework

Compare performance across models, prompts, and datasets:

```bash
python cross_validation.py \
  --datasets very-simple-family-sentences-evals.json family-sentences-evals.json \
  --models sonnet gpt4o \
  --prompt-variations prompt_template.txt prompt_hierarchical.txt \
  --folds 5 \
  --max-sentences 50
```

Options:
- `--datasets`: One or more datasets to evaluate
- `--models`: Specific models to evaluate
- `--model-group`: Predefined group of models (claude, openai, gemini, deepseek)
- `--prompt-variations`: Different prompt templates to compare
- `--folds`: Number of folds for cross-validation
- `--max-sentences`: Maximum sentences to evaluate per fold
- `--output-dir`: Directory to store results

### Dataset Splitting

Split your data for training, validation, and testing:

```bash
python dataset_splitter.py \
  family-sentences-evals.json \
  --output-dir test_split \
  --train-ratio 0.7 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --random-seed 42
```

### Prompt Refinement Workflow

Automatically refine prompts through iterative testing:

```bash
python run_prompt_refinement.py \
  --base-prompt prompt_template.txt \
  --dataset family-sentences-evals.json \
  --output-dir prompt_refinement_workflow \
  --model sonnet \
  --max-iterations 5
```

## Understanding Results

### Output Directory Structure

Results are saved in timestamped directories under `eval-output/`:

```
eval-output/
└── eval-sonnet-claude-20250313-143045/
    ├── README.md                # Overview of results
    ├── extraction_results.json  # Complete results with extracted data
    ├── field_accuracy.csv       # Detailed field-level accuracy
    ├── metadata.json            # Run configuration details
    ├── prompt_improvements.md   # Suggested prompt improvements
    ├── prompt_template.txt      # Copy of the prompt used
    ├── scores.csv               # Sentence-level scores
    └── summary.csv              # Overall metrics summary
```

### Key Metrics

- **Overall Value Accuracy**: Percentage of correctly extracted values
- **Field Precision**: Percentage of extracted fields that are correct
- **Field Recall**: Percentage of expected fields that were extracted
- **Field F1 Score**: Harmonic mean of precision and recall
- **Relationship Accuracy**: Accuracy of relationship identification
- **Name Accuracy**: Accuracy of name extraction

### Optimization Performance Metrics

When running optimized evaluations, additional metrics are recorded:

- **Cache Hit Ratio**: Percentage of cached responses
- **Average Processing Time**: Time per sentence
- **Parallel Efficiency**: Measure of parallel processing gains
- **Selected Tiers**: Distribution of selected prompt tiers

## Troubleshooting

### API Key Issues

If you encounter authentication errors:
1. Verify that environment variables are set correctly
2. Check API key validity and permissions
3. Ensure you have sufficient quota/credits

### Rate Limiting

If experiencing rate limiting errors:
1. Reduce the number of parallel workers
2. Enable the adaptive rate limiter: `--rate-limiting`
3. Process in smaller batches

### Memory Issues

For large datasets causing memory problems:
1. Process in smaller batches with `--batch-size 5`
2. Limit concurrent processing with `--max-workers 3`
3. Use `--max-sentences` to process subsets of data

### Clearing the Cache

If cached responses cause issues:

```python
# Clear all cached responses
from sentence_extraction.optimizations import ResponseCache
ResponseCache("./cache").clear_all()

# Or clear only expired entries
ResponseCache("./cache").clear_expired()
```

## Example Workflows

### Model Comparison

Compare performance across different models:

```bash
python run_model_comparison.py \
  --prompt-file prompt_template.txt \
  --eval-data example_evaluation_data.json \
  --models sonnet opus haiku gpt4o gpt4 \
  --max-sentences 20
```

### Prompt Structure Testing

Test different prompt structures:

```bash
python test_prompt_structure.py \
  --flat-prompt prompt_flat.txt \
  --hierarchical-prompt prompt_hierarchical.txt \
  --eval-data structure_test_dataset.json \
  --model sonnet
```

### Domain-Specific Extraction

Test extraction with domain-specific context:

```bash
python test_domain_context.py \
  --domain-prompt test_domain_prompt.txt \
  --standard-prompt prompt_template.txt \
  --eval-data domain_specific_test.json \
  --model sonnet
```

## Best Practices

1. **Start small**: Test with small datasets before running large evaluations
2. **Consistent temperatures**: Use temperature 0.0 for reproducible results
3. **Dataset quality**: Ensure evaluation data has accurate ground truth
4. **Prompt evolution**: Iteratively refine prompts based on failure analysis
5. **Batch appropriately**: Use batch sizes appropriate for your hardware
6. **Cache responses**: Enable caching for repeated evaluations
7. **Compare systematically**: Use cross-validation for fair comparisons
8. **Document everything**: Keep track of prompt variations and results
9. **Analyze failures**: Focus on understanding extraction errors

## Additional Resources

- **Prompt Refinement Guide**: See `docs/prompt_refinement_guide.md`
- **Optimization Usage Guide**: See `docs/optimization_usage_guide.md`
- **Development Setup**: See `docs/dev_setup.md`

## Getting Help

For development questions, see:
- `docs/dev_setup.md` for development environment setup
- Package code comments for detailed implementation notes
- File an issue on the GitHub repository for bug reports