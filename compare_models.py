#\!/usr/bin/env python3
"""
Script to compare Claude 3.7 and GPT-4o using the optimized prompt.
"""

import os
import json
import subprocess
import datetime
from pathlib import Path

def run_evaluation(prompt_path, model, dataset_path):
    """Run a single evaluation using the evaluation script."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create the comparative results directory
    output_dir = Path("comparative_cv_results")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare command
    cmd = [
        "python", "claude_extraction_eval.py",
        "--model", model,
        "--prompt-file", prompt_path,
        "--eval-data", dataset_path
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
            if "Value accuracy (total score)" in line:
                metrics["value_accuracy"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Field extraction rate" in line:
                metrics["field_extraction_rate"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Average sentence score" in line:
                metrics["avg_sentence_score"] = float(line.split(":")[1].strip().rstrip("%"))
            elif "Successful extractions" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    success_part = parts[1].strip()
                    if "(" in success_part:
                        metrics["success_rate"] = float(success_part.split("(")[1].split("%")[0])
            elif "Processing rate" in line:
                metrics["processing_rate"] = float(line.split(":")[1].strip().split()[0])
        
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return {}

def main():
    # Prompt path
    prompt_path = "prompt_optimized.txt"
    
    # Test dataset
    dataset_path = "family-sentences-evals.json"
    
    # Define models to test
    models = {
        "claude3.7": "sonnet",
        "gpt4o": "gpt4o"
    }
    
    # Store results
    results = {}
    
    # Run the comparison for each model
    for model_name, model_id in models.items():
        print(f"\n=== Testing {model_name} ===")
        metrics = run_evaluation(prompt_path, model_id, dataset_path)
        results[model_name] = metrics
    
    # Compare results
    print("\n=== MODELS COMPARISON ===")
    print(f"Prompt: {prompt_path}")
    print(f"Dataset: {dataset_path}")
    
    metrics_to_display = ["value_accuracy", "field_extraction_rate", "avg_sentence_score", "success_rate", "processing_rate"]
    metric_labels = {
        "value_accuracy": "Value Accuracy", 
        "field_extraction_rate": "Field Extraction Rate", 
        "avg_sentence_score": "Average Sentence Score", 
        "success_rate": "Success Rate",
        "processing_rate": "Processing Rate (sentences/min)"
    }
    
    # Print comparison table
    print("\n| Metric | Claude 3.7 | GPT-4o |")
    print("|--------|-----------|--------|")
    
    for metric in metrics_to_display:
        claude_val = results.get("claude3.7", {}).get(metric, "N/A")
        gpt_val = results.get("gpt4o", {}).get(metric, "N/A")
        
        if metric != "processing_rate" and not isinstance(claude_val, str) and not isinstance(gpt_val, str):
            # Format as percentage
            print(f"| {metric_labels[metric]} | {claude_val:.2f}% | {gpt_val:.2f}% |")
        else:
            # Format as is
            print(f"| {metric_labels[metric]} | {claude_val} | {gpt_val} |")
    
    # Save results to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f"comparative_cv_results/model_comparison_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "prompt": prompt_path,
        "dataset": dataset_path,
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    # Create markdown summary
    md_output = f"comparative_cv_results/model_comparison_{timestamp}.md"
    with open(md_output, "w") as f:
        f.write(f"# Claude 3.7 vs GPT-4o Comparison\n\n")
        f.write(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Prompt:** {prompt_path}\n\n")
        f.write(f"**Dataset:** {dataset_path}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Metric | Claude 3.7 | GPT-4o | Difference |\n")
        f.write("|--------|-----------|--------|------------|\n")
        
        for metric in metrics_to_display:
            claude_val = results.get("claude3.7", {}).get(metric, "N/A")
            gpt_val = results.get("gpt4o", {}).get(metric, "N/A")
            
            if not isinstance(claude_val, str) and not isinstance(gpt_val, str):
                diff = claude_val - gpt_val
                diff_str = f"{diff:+.2f}" + ("%" if metric != "processing_rate" else "")
                
                if metric != "processing_rate":
                    # Format as percentage
                    f.write(f"| {metric_labels[metric]} | {claude_val:.2f}% | {gpt_val:.2f}% | {diff_str} |\n")
                else:
                    # Format as is
                    f.write(f"| {metric_labels[metric]} | {claude_val:.2f} | {gpt_val:.2f} | {diff_str} |\n")
            else:
                f.write(f"| {metric_labels[metric]} | {claude_val} | {gpt_val} | N/A |\n")
        
        f.write("\n## Summary\n\n")
        
        # Calculate which model performed better overall
        claude_score = sum(v for k, v in results.get("claude3.7", {}).items() 
                         if k in ["value_accuracy", "field_extraction_rate", "avg_sentence_score", "success_rate"])
        gpt_score = sum(v for k, v in results.get("gpt4o", {}).items() 
                       if k in ["value_accuracy", "field_extraction_rate", "avg_sentence_score", "success_rate"])
        
        if claude_score > gpt_score:
            f.write("**Claude 3.7** performed better overall in extraction accuracy metrics.\n\n")
        elif gpt_score > claude_score:
            f.write("**GPT-4o** performed better overall in extraction accuracy metrics.\n\n")
        else:
            f.write("Both models performed similarly overall.\n\n")
        
        # Processing speed comparison
        claude_speed = results.get("claude3.7", {}).get("processing_rate", 0)
        gpt_speed = results.get("gpt4o", {}).get("processing_rate", 0)
        
        if not isinstance(claude_speed, str) and not isinstance(gpt_speed, str):
            if claude_speed > gpt_speed:
                f.write(f"**Claude 3.7** was faster, processing {claude_speed:.2f} sentences/minute compared to {gpt_speed:.2f} for GPT-4o.\n")
            elif gpt_speed > claude_speed:
                f.write(f"**GPT-4o** was faster, processing {gpt_speed:.2f} sentences/minute compared to {claude_speed:.2f} for Claude 3.7.\n")
            else:
                f.write(f"Both models processed sentences at the same rate ({claude_speed:.2f} sentences/minute).\n")
    
    print(f"Markdown summary saved to {md_output}")

if __name__ == "__main__":
    main()
