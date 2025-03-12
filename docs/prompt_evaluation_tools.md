# Prompt Evaluation Tools

This document provides an overview of the various Python programs used for prompt evaluation in the sentence extraction project.

## 1. Core Evaluation Framework

### claude_extraction_eval.py
- **Purpose**: Core evaluation script that tests a model's ability to extract structured information from sentences
- **Arguments**: 
  - `--prompt-file`: Template for model prompts
  - `--eval-data`: JSON dataset with sentences and expected extractions
  - `--model`: Model to use (supports Claude, OpenAI, Gemini, DeepSeek families)
  - `--batch-size`: Number of sentences per batch
  - `--max-sentences`: Limit evaluation to this many sentences
  - `--temperature`: Sampling temperature for model responses
- **Output**: Comprehensive evaluation results in JSON format, metrics CSV files, visualizations, and improvement recommendations
- **Role**: Foundation for all evaluation workflows; used by other scripts for comparative testing

## 2. Cross-Validation Framework

### cross_validation.py
- **Purpose**: Framework for comparing prompt variations across models and datasets
- **Arguments**: 
  - `--datasets`: List of datasets to evaluate
  - `--prompt-variations`: Different prompt files to compare
  - `--model-group`: Predefined group of models to test
  - `--models`: Specific models to evaluate
  - `--folds`: Number of cross-validation folds
- **Output**: Statistical comparison of prompt performance across models/datasets, visualizations, and HTML reports
- **Role**: Enables systematic comparison of different prompt approaches with statistical validity

### run_cross_validation_test.py & test_cross_validation.py
- **Purpose**: Simple test wrappers for the cross-validation framework
- **Output**: Basic validation that cross-validation is working properly
- **Role**: Testing and debugging the cross-validation system

### run_limited_cv_test.py
- **Purpose**: Runs a smaller cross-validation test with limited sentences
- **Role**: Quick testing of cross-validation with reduced computational resources

## 3. Prompt Refinement Pipeline

### prompt_refiner.py
- **Purpose**: Implements iterative prompt refinement based on failure analysis
- **Arguments**:
  - Base prompt file path
  - Training dataset path
  - Validation dataset path
  - Various configuration options
- **Output**: Series of refined prompts, performance metrics, and improvement summaries
- **Role**: Core engine for automatically improving prompts through iterative refinement

### failure_analyzer.py
- **Purpose**: Analyzes extraction failures to identify patterns and improvement areas
- **Arguments**: 
  - Results file from evaluation
  - Configuration options for analysis
- **Output**: Detailed analysis reports with improvement recommendations
- **Role**: Critical component of the refinement workflow; identifies what to improve

### run_prompt_refinement.py
- **Purpose**: Coordinates end-to-end prompt refinement workflow
- **Arguments**:
  - Base prompt file path
  - Dataset to use
  - Configuration options for refinement process
- **Output**: Complete refinement workflow with dataset splits, progressive improvement tracking, and final performance comparison
- **Role**: Main orchestrator for a full prompt refinement cycle

### create_refined_prompt.py
- **Purpose**: Utility for creating new prompt variations based on refinement suggestions
- **Role**: Helper tool for the refinement workflow

## 4. Specialized Testing Scripts

### test_domain_context.py
- **Purpose**: Tests if adding domain context before instructions improves extraction
- **Arguments**: Dataset, models, and configuration options
- **Output**: Cross-validation comparison between prompts with/without domain context
- **Role**: Specialized test for understanding the impact of domain context on performance

### test_reasoning_scaffolds.py
- **Purpose**: Tests if chain-of-thought reasoning scaffolds improve extraction
- **Output**: Detailed comparison between standard and CoT prompts with analysis
- **Role**: Evaluates the effectiveness of explicit reasoning steps in prompts

### test_prompt_structure.py
- **Purpose**: Compares different structural approaches to prompt organization
- **Output**: Analysis of how prompt structure affects extraction performance
- **Role**: Helps identify optimal prompt organization strategies

### test_complex_cot.py & simple_cot_test.py
- **Purpose**: Tests chain-of-thought approaches on varying complexity levels
- **Output**: Performance metrics for CoT approaches on different complexity datasets
- **Role**: Examines how reasoning scaffolds perform across complexity levels

### test_baseline_vs_refined.py
- **Purpose**: Direct comparison between baseline and refined prompts
- **Output**: Performance metrics and statistical analysis of improvements
- **Role**: Validates that refinement process actually improves performance

## 5. Model Comparison Tools

### compare_models.py & run_model_comparison.py
- **Purpose**: Compare extraction performance across different models using the same prompts
- **Output**: Comparative metrics, visualizations, and analysis
- **Role**: Evaluates which models perform best for extraction tasks

### life_events_eval.py
- **Purpose**: Specialized evaluation for life events extraction
- **Output**: Performance metrics focused on temporal and biographical information
- **Role**: Domain-specific evaluation tool for life events extraction

## Workflow Integration

The programs integrate in a hierarchical workflow:

1. `claude_extraction_eval.py` serves as the foundation for evaluating any single prompt-model combination
2. `cross_validation.py` builds on this to compare multiple prompts systematically
3. `failure_analyzer.py` examines evaluation results to identify improvement areas
4. `prompt_refiner.py` uses these insights to create better prompts
5. `run_prompt_refinement.py` orchestrates the entire refinement cycle

Specialized testing scripts focus on specific prompt design questions (domain context, reasoning scaffolds, structure) while using the same underlying evaluation framework.

The progressive refinement workflow is the most comprehensive approach, providing a systematic method to improve prompts through data-driven iteration and validation.