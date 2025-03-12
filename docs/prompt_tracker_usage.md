# Prompt Tracker Usage Guide

The Prompt Tracker is a tool for systematically tracking prompt enhancements and test results over time. It provides a centralized system for logging prompt versions, their performance metrics, and evaluation history.

## Installation

Ensure you have the required dependencies:

```bash
pip install pandas matplotlib tabulate
```

## Getting Started

The Prompt Tracker is a command-line tool with several subcommands for different operations.

### Directory Structure

When you run the tool for the first time, it creates the following directory structure:

```
prompt-tracking/
├── prompts/              # Stores copies of all registered prompts
├── evaluations/          # Stores evaluation results
├── prompt_registry.json  # Master registry of all prompts with metadata
├── performance_history.json  # Historical performance by prompt/model
└── refinement_lineage.json   # Tracks prompt refinement history
```

## Commands

### Register a Prompt

Register a new prompt or a refinement of an existing prompt:

```bash
./prompt_tracker.py register path/to/prompt.txt --name "My Prompt" --description "Description of the prompt" --tags "tag1,tag2"
```

If this is a refinement of an existing prompt, you can specify the parent prompt ID:

```bash
./prompt_tracker.py register path/to/refined_prompt.txt --name "Refined Prompt" --parent prompt_20250312_abcd1234 --description "Improved version with better performance"
```

### Log Evaluation Results

After evaluating a prompt, log the results:

```bash
./prompt_tracker.py log prompt_20250312_abcd1234 path/to/results.json --model "claude-3-7-sonnet" --dataset "family-sentences-evals" --notes "Performance on complex sentences"
```

The tool will extract metrics from the results file (if in a recognized format) and update the prompt's performance history.

### List Registered Prompts

View all registered prompts, optionally filtered:

```bash
# List all prompts
./prompt_tracker.py list-prompts

# Filter by tag
./prompt_tracker.py list-prompts --tag "enhanced"

# Filter by minimum accuracy
./prompt_tracker.py list-prompts --min-accuracy 0.85

# Limit the number of results
./prompt_tracker.py list-prompts --limit 5
```

### List Evaluations

View all evaluations, optionally filtered:

```bash
# List all evaluations
./prompt_tracker.py list-evals

# Filter by prompt ID
./prompt_tracker.py list-evals --prompt prompt_20250312_abcd1234

# Filter by model
./prompt_tracker.py list-evals --model "claude-3-7-sonnet"

# Filter by dataset
./prompt_tracker.py list-evals --dataset "family-sentences-evals"

# Limit the number of results
./prompt_tracker.py list-evals --limit 10
```

### Find Best Performing Prompts

Identify the best performing prompts:

```bash
# Get the top 5 best performing prompts
./prompt_tracker.py best

# Filter by model
./prompt_tracker.py best --model "claude-3-7-sonnet"

# Filter by dataset
./prompt_tracker.py best --dataset "family-sentences-evals"

# Change the number of results
./prompt_tracker.py best --limit 10
```

### Plot Performance Trend

Generate a plot of prompt performance over time:

```bash
# Plot all prompts
./prompt_tracker.py plot

# Plot specific prompts
./prompt_tracker.py plot --prompts "prompt_id1,prompt_id2"

# Filter by model
./prompt_tracker.py plot --model "claude-3-7-sonnet"

# Save the plot to a file
./prompt_tracker.py plot --output "performance_trend.png"
```

### Generate a Comprehensive Report

Create a markdown report of prompt performance:

```bash
# Display the report
./prompt_tracker.py report

# Save the report to a file
./prompt_tracker.py report --output "prompt_performance_report.md"
```

## Workflow Example

Here's a typical workflow for using the Prompt Tracker:

1. Register your baseline prompt:
   ```bash
   ./prompt_tracker.py register prompt_template.txt --name "Baseline" --tags "baseline"
   ```

2. Evaluate the prompt with your evaluation script and log the results:
   ```bash
   python claude_extraction_eval.py --prompt-file prompt_template.txt --eval-data family-sentences-evals.json --model claude-3-7-sonnet
   ./prompt_tracker.py log prompt_20250312_abcd1234 eval-output/extraction_results.json --model "claude-3-7-sonnet" --dataset "family-sentences-evals"
   ```

3. Create a refined prompt and register it:
   ```bash
   ./prompt_tracker.py register prompt_optimized.txt --name "Enhanced Hierarchical" --parent prompt_20250312_abcd1234 --tags "enhanced,hierarchical"
   ```

4. Evaluate the refined prompt and log the results:
   ```bash
   python claude_extraction_eval.py --prompt-file prompt_optimized.txt --eval-data family-sentences-evals.json --model claude-3-7-sonnet
   ./prompt_tracker.py log prompt_20250313_efgh5678 eval-output/extraction_results.json --model "claude-3-7-sonnet" --dataset "family-sentences-evals"
   ```

5. Compare performance and find the best prompt:
   ```bash
   ./prompt_tracker.py plot --prompts "prompt_20250312_abcd1234,prompt_20250313_efgh5678"
   ./prompt_tracker.py best
   ```

6. Generate a report:
   ```bash
   ./prompt_tracker.py report --output "prompt_performance_report.md"
   ```

## Integration with Existing Tools

The Prompt Tracker is designed to work with the existing evaluation pipeline:

1. Use `claude_extraction_eval.py` for evaluation as usual
2. Register prompts and log results with Prompt Tracker
3. Use Prompt Tracker to analyze performance across different prompt versions and models

This provides a systematic way to track prompt effectiveness and improvements over time.