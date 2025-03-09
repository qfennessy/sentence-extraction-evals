#!/usr/bin/env python3
"""
Test script to evaluate if Claude performs better with hierarchical vs. flat instruction structures.

This script runs a cross-validation comparison between a flat instruction structure
and a hierarchical instruction structure to determine which produces better
extraction results.
"""

import os
import json
import datetime
import argparse
from pathlib import Path
from cross_validation import CrossValidator

def main():
    """Run a structure preference analysis test."""
    parser = argparse.ArgumentParser(
        description="Test if Claude performs better with hierarchical vs. flat instruction structure"
    )
    parser.add_argument(
        "--dataset", 
        default="very-simple-family-sentences-evals.json",
        help="Dataset file to evaluate (default: very-simple-family-sentences-evals.json)"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["sonnet"],
        help="Models to test (default: sonnet)"
    )
    parser.add_argument(
        "--folds", 
        type=int, 
        default=3,
        help="Number of folds for cross-validation (default: 3)"
    )
    parser.add_argument(
        "--max-sentences", 
        type=int, 
        default=5,
        help="Maximum sentences to evaluate per fold (default: 5)"
    )
    parser.add_argument(
        "--output-dir", 
        default="structure_preference_results",
        help="Directory to store results"
    )
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) / f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup cross-validator
    cv = CrossValidator(
        datasets=[args.dataset],
        models=args.models,
        folds=args.folds,
        max_sentences=args.max_sentences,
        output_dir=output_dir,
        base_prompt_path="prompt_template.txt",
        temperature=0.0,
    )
    
    # Add prompt variations
    cv.add_prompt_variation(
        "hierarchical", 
        "prompt_hierarchical.txt",
        "Hierarchical instruction structure with nested sections and Markdown formatting"
    )
    cv.add_prompt_variation(
        "flat", 
        "prompt_flat.txt",
        "Flat instruction structure with minimal whitespace and linear instructions"
    )
    
    # Write experiment description
    experiment_details = {
        "title": "Structure Preference Analysis: Hierarchical vs. Flat Instructions",
        "description": """
This experiment evaluates whether Claude performs better with hierarchical or flat instruction structures.
Two prompt variations were tested:

1. **Hierarchical Structure**: Uses multiple levels of organization with headings/subheadings, visual 
   separation, and markdown-style formatting to clearly delineate sections, tasks, and guidelines.

2. **Flat Structure**: Presents the same content in a linear format with minimal whitespace, 
   no hierarchical organization, and condensed instructions.

Both prompts contain identical information and requirements, differing only in their structural
organization and formatting.
        """,
        "dataset": args.dataset,
        "models": args.models,
        "folds": args.folds,
        "max_sentences": args.max_sentences,
        "date": timestamp,
    }
    
    with open(output_dir / "experiment_details.json", 'w') as f:
        json.dump(experiment_details, f, indent=2)
    
    # Run cross-validation
    print(f"Starting structure preference analysis test...")
    print(f"Comparing hierarchical vs. flat prompt structures")
    print(f"Using dataset: {args.dataset}")
    print(f"Testing models: {', '.join(args.models)}")
    print(f"Running {args.folds} folds with max {args.max_sentences} sentences per fold")
    
    results = cv.run_cross_validation()
    
    # Create summary report
    create_summary_report(results, output_dir)
    
    print(f"Test completed. Results saved to {output_dir}")

def create_summary_report(results, output_dir):
    """Create a detailed summary report of the test results."""
    # Extract key metrics from results
    summary = {
        "flat": {},
        "hierarchical": {},
        "difference": {},
    }
    
    for result in results:
        prompt_type = result["prompt"]
        model = result["model"]
        
        if prompt_type in ["flat", "hierarchical"]:
            agg = result["aggregate_metrics"]
            
            # Initialize model in summary if not present
            for section in summary.values():
                if model not in section:
                    section[model] = {}
            
            # Add metrics
            metrics = {
                "value_accuracy": agg["value_accuracy"]["mean"] if "value_accuracy" in agg else 0.0,
                "field_extraction_rate": agg["field_extraction_rate"]["mean"] if "field_extraction_rate" in agg else 0.0,
                "avg_sentence_score": agg["avg_sentence_score"]["mean"] if "avg_sentence_score" in agg else 0.0,
                "success_rate": agg["success_rate"]["mean"] if "success_rate" in agg else 0.0,
            }
            summary[prompt_type][model] = metrics
            
    # Calculate differences (in a separate loop after all metrics are collected)
    for model in summary["hierarchical"]:
        if model in summary["flat"]:
            summary["difference"][model] = {}
            for metric in summary["hierarchical"][model]:
                flat_val = summary["flat"][model][metric]
                hier_val = summary["hierarchical"][model][metric]
                summary["difference"][model][metric] = hier_val - flat_val
    
    # Write summary to JSON
    with open(output_dir / "structure_preference_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create markdown report
    md_report = """# Structure Preference Analysis Results

## Overview
This report compares the performance of Claude models using hierarchical vs. flat instruction structures
for information extraction tasks. The same information was presented in both formats, with only the
structural organization differing between the two prompt variations.

## Key Findings

"""
    
    # Add findings section based on results
    for model in summary["difference"]:
        md_report += f"### Model: {model}\n\n"
        md_report += "| Metric | Hierarchical | Flat | Difference |\n"
        md_report += "| ------ | ------------ | ---- | ---------- |\n"
        
        for metric in summary["difference"][model]:
            hier_val = summary["hierarchical"][model][metric]
            flat_val = summary["flat"][model][metric]
            diff = summary["difference"][model][metric]
            
            # Format values
            hier_val_fmt = f"{hier_val:.2f}%"
            flat_val_fmt = f"{flat_val:.2f}%"
            
            # Format difference with color indicator
            if diff > 0:
                diff_fmt = f"+{diff:.2f}% 🟢"
            elif diff < 0:
                diff_fmt = f"{diff:.2f}% 🔴"
            else:
                diff_fmt = f"{diff:.2f}% ⚪"
                
            # Format metric name
            metric_name = " ".join(word.capitalize() for word in metric.split("_"))
            
            md_report += f"| {metric_name} | {hier_val_fmt} | {flat_val_fmt} | {diff_fmt} |\n"
        
        md_report += "\n"
    
    # Add conclusion based on overall results
    overall_diff = 0
    metric_count = 0
    
    for model in summary["difference"]:
        for metric in summary["difference"][model]:
            overall_diff += summary["difference"][model][metric]
            metric_count += 1
    
    avg_diff = overall_diff / metric_count if metric_count > 0 else 0
    
    if avg_diff > 2:
        conclusion = """## Conclusion
The results show that **hierarchical instruction structure significantly outperforms flat structure** 
for information extraction tasks. Models appear to better understand, process, and follow instructions 
when they are organized in a clear, visually separated hierarchical format with headings and nested sections.

### Recommendations
- Use hierarchical formatting with clear section headings for complex extraction tasks
- Incorporate visual structure (whitespace, indentation) to separate logical sections
- Consider using markdown-style formatting for enhanced readability
- Organize related guidelines into nested hierarchical groups"""
    elif avg_diff > 0.5:
        conclusion = """## Conclusion
The results suggest that **hierarchical instruction structure slightly outperforms flat structure**
for information extraction tasks. While the difference is not dramatic, there appears to be a 
consistent advantage to using structured, visually organized prompts.

### Recommendations
- Consider using hierarchical formatting for complex extraction tasks
- Incorporate some visual structure (whitespace, section breaks) even in simpler prompts
- Group related instructions together with clear headings"""
    elif avg_diff < -0.5:
        conclusion = """## Conclusion
Surprisingly, the results suggest that **flat instruction structure slightly outperforms hierarchical structure**
for information extraction tasks. This may indicate that for this specific task type, conciseness and
linear presentation of instructions is more effective.

### Recommendations
- Consider using more compact, linear formatting for extraction tasks
- Avoid excessive hierarchy and nesting that might dilute key instructions
- Test both approaches for your specific use case as results may vary"""
    else:
        conclusion = """## Conclusion
The results show **no significant difference between hierarchical and flat instruction structures**
for information extraction tasks. Claude appears to effectively process and follow instructions
regardless of their structural organization.

### Recommendations
- Choose the format that best suits your personal preference or organizational standards
- Consider other factors (like prompt length, example quality, instruction clarity) that may
  have more impact on performance than structural formatting
- Test both approaches for your specific use case to confirm these general findings"""
    
    md_report += conclusion
    
    # Write report
    with open(output_dir / "structure_preference_report.md", 'w') as f:
        f.write(md_report)

if __name__ == "__main__":
    main()