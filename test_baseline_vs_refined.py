#!/usr/bin/env python3
"""
Compare baseline and refined prompt performance on the test set
"""

import os
import subprocess
from pathlib import Path

# Set paths
base_prompt = "prompt_template.txt"
refined_prompt = "refined_prompt_simple_test.txt"
dataset_name = "family-sentences-evals"
test_path = f"simple_test_splits/{dataset_name}/{dataset_name}_test.json"

# Function to extract metrics from evaluation output
def extract_metrics(stdout):
    metrics = {}
    for line in stdout.split("\n"):
        if "Value accuracy" in line:
            metrics["value_accuracy"] = float(line.split(":")[1].strip().rstrip("%"))
        elif "Field extraction rate" in line:
            metrics["field_extraction_rate"] = float(line.split(":")[1].strip().rstrip("%"))
        elif "Average sentence score" in line:
            metrics["avg_sentence_score"] = float(line.split(":")[1].strip().rstrip("%"))
    return metrics

# Step 1: Evaluate baseline prompt on test set
print("Evaluating baseline prompt on test set...")
baseline_cmd = [
    "python", "claude_extraction_eval.py",
    "--model", "sonnet",
    "--prompt-file", base_prompt,
    "--eval-data", test_path
]

baseline_result = subprocess.run(baseline_cmd, capture_output=True, text=True, check=True)
print(baseline_result.stdout)
baseline_metrics = extract_metrics(baseline_result.stdout)

# Step 2: Evaluate refined prompt on test set
print("\nEvaluating refined prompt on test set...")
refined_cmd = [
    "python", "claude_extraction_eval.py",
    "--model", "sonnet",
    "--prompt-file", refined_prompt,
    "--eval-data", test_path
]

refined_result = subprocess.run(refined_cmd, capture_output=True, text=True, check=True)
print(refined_result.stdout)
refined_metrics = extract_metrics(refined_result.stdout)

# Step 3: Compare results
print("\n========== COMPARISON RESULTS ==========")
print(f"{'Metric':<25} {'Baseline':<15} {'Refined':<15} {'Improvement':<15}")
print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")

for metric in ["value_accuracy", "field_extraction_rate", "avg_sentence_score"]:
    baseline_value = baseline_metrics.get(metric, 0)
    refined_value = refined_metrics.get(metric, 0)
    improvement = refined_value - baseline_value
    print(f"{metric:<25} {baseline_value:<15.2f} {refined_value:<15.2f} {improvement:+<15.2f}")

if refined_metrics.get("value_accuracy", 0) > baseline_metrics.get("value_accuracy", 0):
    print("\n✅ Refined prompt shows improvement!")
else:
    print("\n❌ Refined prompt did not show significant improvement.")