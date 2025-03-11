#!/usr/bin/env python3
"""
Run Prompt Refinement Process

This script coordinates the end-to-end prompt refinement workflow:
1. Split a dataset into train, validation, and test sets
2. Analyze extraction failures on the training set
3. Refine the prompt based on failure patterns
4. Validate refinements on the validation set
5. Evaluate the final prompt on the holdout test set
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
import datetime


def run_workflow(
    base_prompt_path: str,
    dataset_path: str,
    output_dir: str = "prompt_refinement_workflow",
    model: str = "sonnet",
    max_iterations: int = 5,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """Run the end-to-end prompt refinement workflow.
    
    Args:
        base_prompt_path: Path to the initial prompt template
        dataset_path: Path to the dataset to use
        output_dir: Directory to store all workflow outputs
        model: Model to use for evaluations
        max_iterations: Maximum number of refinement iterations
        train_ratio: Ratio of data for training
        validation_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        random_seed: Random seed for reproducibility
    """
    # Create workflow directory
    workflow_dir = Path(output_dir)
    os.makedirs(workflow_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = workflow_dir / f"run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Set up subdirectories
    splits_dir = run_dir / "dataset_splits"
    analysis_dir = run_dir / "failure_analysis"
    refinement_dir = run_dir / "prompt_refinement"
    final_eval_dir = run_dir / "final_evaluation"
    
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(refinement_dir, exist_ok=True)
    os.makedirs(final_eval_dir, exist_ok=True)
    
    # Step 1: Split the dataset
    print("\n=== Step 1: Splitting Dataset ===\n")
    
    split_cmd = [
        "python", "dataset_splitter.py",
        dataset_path,
        "--output-dir", str(splits_dir),
        "--train-ratio", str(train_ratio),
        "--validation-ratio", str(validation_ratio),
        "--test-ratio", str(test_ratio),
        "--random-seed", str(random_seed)
    ]
    
    try:
        split_result = subprocess.run(split_cmd, capture_output=True, text=True, check=True)
        print(split_result.stdout)
        
        # Extract split paths from output
        dataset_name = Path(dataset_path).stem
        train_path = splits_dir / dataset_name / f"{dataset_name}_train.json"
        validation_path = splits_dir / dataset_name / f"{dataset_name}_validation.json"
        test_path = splits_dir / dataset_name / f"{dataset_name}_test.json"
        
        # Verify that splits were created
        if not (train_path.exists() and validation_path.exists() and test_path.exists()):
            print("Error: Dataset splits were not created correctly")
            return
            
    except subprocess.CalledProcessError as e:
        print(f"Error splitting dataset: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return
    
    # Step 2: Run prompt refinement process
    print("\n=== Step 2: Running Prompt Refinement ===\n")
    
    refine_cmd = [
        "python", "prompt_refiner.py",
        str(base_prompt_path),
        str(train_path),
        str(validation_path),
        "--output-dir", str(refinement_dir),
        "--max-iterations", str(max_iterations),
        "--model", model
    ]
    
    try:
        refine_result = subprocess.run(refine_cmd, capture_output=True, text=True, check=True)
        print(refine_result.stdout)
        
        # Extract best prompt path from output
        best_prompt_path = None
        for line in refine_result.stdout.split("\n"):
            if "Best prompt:" in line:
                best_prompt_path = line.split("Best prompt:")[1].strip()
                break
                
        if not best_prompt_path or not os.path.exists(best_prompt_path):
            print("Could not identify best prompt from refinement process")
            # Fall back to base prompt
            best_prompt_path = base_prompt_path
            
    except subprocess.CalledProcessError as e:
        print(f"Error running prompt refinement: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        # Fall back to base prompt
        best_prompt_path = base_prompt_path
    
    # Step 3: Evaluate on test set
    print("\n=== Step 3: Evaluating on Test Set ===\n")
    
    test_cmd = [
        "python", "claude_extraction_eval.py",
        "--model", model,
        "--prompt-file", best_prompt_path,
        "--eval-data", str(test_path)
    ]
    
    try:
        test_result = subprocess.run(test_cmd, capture_output=True, text=True, check=True)
        print(test_result.stdout)
        
        # Extract key metrics from output
        test_metrics = {}
        for line in test_result.stdout.split("\n"):
            if "Value accuracy" in line:
                test_metrics["value_accuracy"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Field extraction rate" in line:
                test_metrics["field_extraction_rate"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Average sentence score" in line:
                test_metrics["avg_sentence_score"] = float(line.split(":")[1].strip().rstrip("%"))
                
        # Find test output directory
        test_output_dir = None
        for line in test_result.stdout.split("\n"):
            if "Output directory:" in line:
                test_output_dir = line.split("Output directory:")[1].strip()
                break
                
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating on test set: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        test_metrics = {}
        test_output_dir = None
    
    # Step 4: Run baseline evaluation on test set using original prompt
    print("\n=== Step 4: Evaluating Baseline on Test Set ===\n")
    
    baseline_cmd = [
        "python", "claude_extraction_eval.py",
        "--model", model,
        "--prompt-file", base_prompt_path,
        "--eval-data", str(test_path)
    ]
    
    try:
        baseline_result = subprocess.run(baseline_cmd, capture_output=True, text=True, check=True)
        print(baseline_result.stdout)
        
        # Extract key metrics from output
        baseline_metrics = {}
        for line in baseline_result.stdout.split("\n"):
            if "Value accuracy" in line:
                baseline_metrics["value_accuracy"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Field extraction rate" in line:
                baseline_metrics["field_extraction_rate"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Average sentence score" in line:
                baseline_metrics["avg_sentence_score"] = float(line.split(":")[1].strip().rstrip("%"))
                
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating baseline on test set: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        baseline_metrics = {}
    
    # Step 5: Generate final report
    print("\n=== Step 5: Generating Final Report ===\n")
    
    report_path = run_dir / "final_report.md"
    
    # Create header with key information
    with open(report_path, 'w') as f:
        f.write("# Prompt Refinement Workflow Results\n\n")
        f.write(f"- **Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Dataset:** {Path(dataset_path).name}\n")
        f.write(f"- **Base prompt:** {Path(base_prompt_path).name}\n")
        f.write(f"- **Model:** {model}\n")
        f.write(f"- **Iterations:** {max_iterations}\n\n")
        
        f.write("## Test Set Performance\n\n")
        
        if test_metrics and baseline_metrics:
            baseline_acc = baseline_metrics.get("value_accuracy", 0)
            refined_acc = test_metrics.get("value_accuracy", 0)
            improvement = refined_acc - baseline_acc
            
            f.write("| Metric | Baseline Prompt | Refined Prompt | Improvement |\n")
            f.write("|--------|----------------|----------------|-------------|\n")
            f.write(f"| Value Accuracy | {baseline_acc:.2f}% | {refined_acc:.2f}% | {improvement:.2f}% |\n")
            
            baseline_field = baseline_metrics.get("field_extraction_rate", 0)
            refined_field = test_metrics.get("field_extraction_rate", 0)
            field_improvement = refined_field - baseline_field
            
            f.write(f"| Field Extraction Rate | {baseline_field:.2f}% | {refined_field:.2f}% | {field_improvement:.2f}% |\n")
            
            baseline_score = baseline_metrics.get("avg_sentence_score", 0)
            refined_score = test_metrics.get("avg_sentence_score", 0)
            score_improvement = refined_score - baseline_score
            
            f.write(f"| Avg Sentence Score | {baseline_score:.2f}% | {refined_score:.2f}% | {score_improvement:.2f}% |\n\n")
        else:
            f.write("Could not calculate performance comparison due to evaluation errors.\n\n")
        
        # Add prompt comparison details
        f.write("## Prompt Comparison\n\n")
        
        baseline_path = Path(base_prompt_path)
        best_path = Path(best_prompt_path)
        
        if baseline_path.exists() and best_path.exists():
            try:
                with open(baseline_path, 'r') as bp:
                    baseline_prompt = bp.read()
                    baseline_length = len(baseline_prompt.split())
                    
                with open(best_path, 'r') as rp:
                    refined_prompt = rp.read()
                    refined_length = len(refined_prompt.split())
                    
                f.write(f"- **Baseline prompt length:** {baseline_length} words\n")
                f.write(f"- **Refined prompt length:** {refined_length} words\n")
                f.write(f"- **Size increase:** {refined_length - baseline_length} words ({(refined_length/baseline_length - 1)*100:.1f}%)\n\n")
                
                # Identify additions (simple approach - check for new section headers)
                baseline_sections = set(section.strip() for section in baseline_prompt.split('#') if section.strip())
                refined_sections = set(section.strip() for section in refined_prompt.split('#') if section.strip())
                
                new_sections = refined_sections - baseline_sections
                
                if new_sections:
                    f.write("### New Sections Added\n\n")
                    for section in new_sections:
                        section_name = section.split('\n')[0] if '\n' in section else section
                        f.write(f"- {section_name}\n")
                    f.write("\n")
                    
                # Check for "ADDITIONAL GUIDANCE" section
                if "ADDITIONAL GUIDANCE FOR SPECIFIC CASES" in refined_prompt:
                    f.write("### Additional Guidance Added\n\n")
                    guidance_section = refined_prompt.split("# ADDITIONAL GUIDANCE FOR SPECIFIC CASES")[1]
                    guidance_topics = [line.strip('# ') for line in guidance_section.split('\n') if line.startswith('## ')]
                    
                    for topic in guidance_topics:
                        f.write(f"- {topic}\n")
                    f.write("\n")
                    
            except Exception as e:
                f.write(f"Error comparing prompts: {str(e)}\n\n")
        else:
            f.write("Could not load prompts for comparison.\n\n")
            
        # Add file references
        f.write("## Files\n\n")
        f.write(f"- **Dataset splits:** {splits_dir}\n")
        f.write(f"- **Refinement outputs:** {refinement_dir}\n")
        f.write(f"- **Final evaluation:** {final_eval_dir}\n")
        f.write(f"- **Baseline prompt:** {base_prompt_path}\n")
        f.write(f"- **Refined prompt:** {best_prompt_path}\n")
    
    print(f"\nWorkflow complete! Final report saved to {report_path}")
    
    # Return key paths and metrics
    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "test_path": str(test_path),
        "best_prompt_path": best_prompt_path,
        "baseline_metrics": baseline_metrics,
        "refined_metrics": test_metrics
    }


def main():
    """Run the workflow from command line."""
    parser = argparse.ArgumentParser(description="Run end-to-end prompt refinement workflow")
    parser.add_argument("base_prompt", help="Path to the base prompt template")
    parser.add_argument("dataset", help="Path to the dataset to use")
    parser.add_argument("--output-dir", default="prompt_refinement_workflow", help="Directory to store all workflow outputs")
    parser.add_argument("--model", default="sonnet", help="Model to use for evaluations")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of refinement iterations")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of data for training")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="Ratio of data for validation")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Ratio of data for testing")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    run_workflow(
        base_prompt_path=args.base_prompt,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        model=args.model,
        max_iterations=args.max_iterations,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()