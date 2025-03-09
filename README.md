# Claude Family Information Extraction Evaluation

This project evaluates Claude's ability to extract structured family information from natural language sentences.

## Overview

The evaluation script tests Claude's capabilities in extracting details such as:
- Family member names
- Family relationships
- Ages
- Genders
- Occupations

## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/sentence-extraction-evals.git
cd sentence-extraction-evals
pip install anthropic pandas tqdm
```

## Setup

Before running the evaluation, you need:

1. An Anthropic API key
2. A prompt template file
3. A JSON evaluation dataset

### API Key

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

Alternatively, you can pass it directly using the `--api-key` argument.

### Prompt Template

Create a prompt template file (see `prompt_template.txt` for an example) that instructs Claude how to extract family information. The template should include a placeholder `{SENTENCE}` that will be replaced with each test sentence.

### Evaluation Dataset

Prepare a JSON file with the following structure:

```json
{
  "sentences": [
    {
      "sentence": "John's sister Mary is a 30-year-old doctor with two kids.",
      "extracted_information": {
        "family_members": [
          {
            "name": "Mary",
            "age": "30",
            "gender": "female",
            "occupation": "doctor",
            "relation_to": [
              {
                "name": "John",
                "relationship": "sister"
              }
            ]
          },
          {
            "name": "John",
            "age": null,
            "gender": "male",
            "occupation": null,
            "relation_to": [
              {
                "name": "Mary",
                "relationship": "brother"
              }
            ]
          }
        ]
      }
    },
    // More test sentences...
  ]
}
```

## Usage

Run the evaluation script:

```bash
python claude_extraction_eval.py --prompt-file prompt_template.txt --eval-data evaluation_data.json
```

### Command Line Options

- `--prompt-file`: Path to the prompt template file (required)
- `--eval-data`: Path to the evaluation data JSON file (required)
- `--api-key`: Anthropic API key (optional if set as environment variable)
- `--model`: Claude model to use (default: "claude-3-5-sonnet-20240620")
- `--output-file`: Output file for results (default: "extraction_results.json")
- `--batch-size`: Number of sentences to evaluate in each batch (default: 5)
- `--max-sentences`: Maximum number of sentences to evaluate (default: all)

## Output

The script generates:

1. A detailed JSON file with all extraction results
2. A CSV summary of extraction performance by field
3. A console output with key metrics

### Example Console Output

```
Loaded 50 sentences for evaluation
Using model: claude-3-5-sonnet-20240620
Processing batch 1: 100%|██████████| 5/5 [00:15<00:00, 3.12s/it]
...

Evaluation Summary:
Total sentences: 50
Successful extractions: 48 (96.00%)
Failed extractions: 2 (4.00%)
Overall field extraction rate: 94.25%

Top 5 best extracted fields:
  name: 98.75% (80 occurrences)
  relation_to: 97.50% (40 occurrences)
  gender: 95.00% (60 occurrences)
  age: 92.31% (39 occurrences)
  occupation: 87.50% (32 occurrences)

Bottom 5 worst extracted fields:
  occupation: 87.50% (32 occurrences)
  age: 92.31% (39 occurrences)
  gender: 95.00% (60 occurrences)
  relation_to: 97.50% (40 occurrences)
  name: 98.75% (80 occurrences)

Detailed results saved to extraction_results.json
```

## Troubleshooting

Common issues:

1. **API Key Problems**: Ensure your API key is correct and has sufficient permissions.
2. **Rate Limiting**: If you encounter rate limiting, try increasing the batch size and adding longer delays.
3. **JSON Parsing Errors**: If Claude's responses cannot be parsed, check your prompt template to ensure it clearly instructs Claude to return only a valid JSON object.

## License

[Include license information here]

## Contact

[Your contact information or how to submit issues]