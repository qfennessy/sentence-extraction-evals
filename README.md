# Family Information Extraction Evaluation Framework

A comprehensive framework for evaluating LLMs' ability to extract structured family information from natural language sentences.

## Overview

This framework evaluates how well language models can extract structured information about family members and relationships from text. It supports:

- Multiple model providers (Anthropic, OpenAI, Google, DeepSeek)
- Comprehensive evaluation metrics
- Cross-validation of prompt variations
- Systematic comparison across model sizes

The system extracts details including:
- Family member names and relationships
- Biographical information (age, occupation, birthplace, etc.)
- Life events and achievements
- Complex family structures

## Installation

```bash
pip install -r requirements.txt
```

## Simple Evaluation

To run a basic evaluation:

```bash
python claude_extraction_eval.py --model sonnet --prompt-file prompt_template.txt --eval-data test-family-sentences-evals.json
```

Options:
- `--model`: Model to use (sonnet, opus, haiku, gpt4o, gpt4, gpt35, pro, ultra, coder, chat)
- `--prompt-file`: Path to the prompt template
- `--eval-data`: Path to the evaluation data
- `--max-sentences`: Maximum number of sentences to evaluate
- `--temperature`: Temperature for model responses (0.0-1.0)

## Cross-Validation Framework

### Purpose
The cross-validation framework enables systematic comparison of:
- Different models (varying in size and provider)
- Different prompt variations
- Performance across multiple datasets with varying complexity

### Usage

Basic cross-validation:

```bash
python cross_validation.py --datasets test-family-sentences-evals.json --model-group claude --prompt-variations prompt_template.txt prompt_variation1.txt --folds 5
```

To compare across different complexities:

```bash
python cross_validation.py --datasets very-simple-family-sentences-evals.json family-sentences-evals.json complex-family-sentences-evals.json --models sonnet gpt4o --prompt-variations prompt_template.txt --max-sentences 10
```

Options:
- `--datasets`: One or more datasets to evaluate
- `--model-group`: Predefined group of models (claude, openai, gemini, deepseek, all_small, all_medium, all_large)
- `--models`: Specific models to evaluate
- `--prompt-variations`: Different prompt templates to compare
- `--folds`: Number of folds for cross-validation
- `--max-sentences`: Maximum sentences to evaluate per fold
- `--output-dir`: Directory to store results

### Results

The framework produces:
- Detailed CSV reports with statistical metrics
- Visualizations comparing performance
- HTML reports for easy interpretation
- Full results in JSON for further analysis

## Dataset Format

Evaluation datasets follow this structure:

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
    }
  ]
}
```

For complex datasets, a nested structure with `family_members` is supported.

## Available Datasets

The repository includes several datasets with varying complexity:

- **very-simple-family-sentences-evals.json**: Basic sentences with single relationships
- **family-sentences-evals.json**: Medium complexity with multiple attributes
- **test-family-sentences-evals.json**: Test set with more complex relationships
- **complex-family-sentences-evals.json**: Highly complex sentences with intricate family structures

## Prompt Engineering

The system uses a prompt template that includes a `{SENTENCE}` placeholder. The current template emphasizes:

1. Structured extraction of all family members
2. Correct relationship representation
3. Complete biographical details
4. Handling of complex family structures

See `prompt_template.txt` for the current implementation.

## Output

Evaluation results are stored in timestamped directories under `eval-output/` and include:

- Extraction results (JSON)
- Field accuracy metrics (CSV)
- Error analysis (TXT)
- Prompt improvement suggestions

## License

See LICENSE file for details.