#!/usr/bin/env python3
"""
Simple direct test to compare prompt variations.
"""

import os
import json
import subprocess
import argparse
from datetime import datetime

def run_evaluation(prompt_path, model, dataset_path, output_dir):
    """Run a single evaluation using the evaluation script."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare command
    cmd = [
        "python", "claude_extraction_eval.py",
        "--model", model,
        "--prompt-file", prompt_path,
        "--eval-data", dataset_path,
        "--temperature", "0.0",
        "--max-sentences", "3"
    ]
    
    # Run evaluation
    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
        # Extract metrics from output
        metrics = {}
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if "Value accuracy" in line:
                metrics["value_accuracy"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Field extraction rate" in line:
                metrics["field_extraction_rate"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Average sentence score" in line:
                metrics["avg_sentence_score"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Successful extractions" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    success_part = parts[1].strip()
                    metrics["success_rate"] = float(success_part.split("(")[1].split("%")[0])
        
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Simple test for prompt variations")
    parser.add_argument("--model", default="sonnet", help="Model to test")
    parser.add_argument("--dataset", default="test_domain_context_dataset.json", help="Dataset file")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = f"test_results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Run tests for each prompt variation
    with_context_metrics = run_evaluation(
        "prompt_with_context.txt", 
        args.model, 
        args.dataset, 
        os.path.join(base_output_dir, "with_context")
    )
    
    without_context_metrics = run_evaluation(
        "prompt_without_context.txt", 
        args.model, 
        args.dataset, 
        os.path.join(base_output_dir, "without_context")
    )
    
    # Compare results
    print("\n=== RESULTS COMPARISON ===")
    metrics = ["value_accuracy", "field_extraction_rate", "avg_sentence_score", "success_rate"]
    
    print("\nWith Domain Context:")
    for metric in metrics:
        print(f"  {metric}: {with_context_metrics.get(metric, 'N/A')}%")
    
    print("\nWithout Domain Context:")
    for metric in metrics:
        print(f"  {metric}: {without_context_metrics.get(metric, 'N/A')}%")
    
    # Save results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "timestamp": timestamp,
        "with_context": with_context_metrics,
        "without_context": without_context_metrics
    }
    
    with open(os.path.join(base_output_dir, "comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {base_output_dir}/comparison.json")

if __name__ == "__main__":
    main()