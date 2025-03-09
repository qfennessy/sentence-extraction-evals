#!/usr/bin/env python3
"""
Simple test script for cross-validation framework.
"""

import os
import json
from pathlib import Path
from cross_validation import CrossValidator

def main():
    """Run a simplified cross-validation test."""
    print("Testing cross-validation framework...")
    
    # Set up a simple test case
    test_dataset = "very-simple-family-sentences-evals.json"
    prompt_path = "prompt_template.txt"
    
    # Create test validator with minimal settings
    cv = CrossValidator(
        datasets=[test_dataset],
        models=["sonnet"],
        folds=2,
        max_sentences=2,
        output_dir="cv_test_results"
    )
    
    # Add prompt variation
    cv.add_prompt_variation("base", prompt_path, "Base Prompt")
    
    # Run individual dataset processing steps to check functionality
    print(f"\nPreparing dataset folds for {test_dataset}...")
    fold_paths = cv.prepare_dataset_folds(test_dataset)
    print(f"Created {len(fold_paths)} folds.")
    
    # Test a single evaluation
    print(f"\nTesting single evaluation with first fold...")
    if fold_paths:
        result = cv.run_evaluation(prompt_path, "sonnet", fold_paths[0])
        print(f"Evaluation result successful: {result['success']}")
        if result['success']:
            print(f"Metrics: {result['metrics']}")
    
    print("\nTest complete.")

if __name__ == "__main__":
    main()