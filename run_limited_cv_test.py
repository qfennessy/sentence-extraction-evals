#!/usr/bin/env python3
"""
Run a very limited test of the cross-validation framework.
Tests just one model on a small dataset with minimal folds.
"""

from cross_validation import CrossValidator

def main():
    """Run limited cross-validation test."""
    print("Running minimal cross-validation test...")
    
    # Test with just one dataset and one model
    cv = CrossValidator(
        datasets=["very-simple-family-sentences-evals.json"],
        models=["sonnet"],
        folds=2,
        max_sentences=2,
        output_dir="mini_cv_results"
    )
    
    # Use the current prompt
    cv.add_prompt_variation("current", "prompt_template.txt", "Current Prompt Template")
    
    # Run full cross-validation
    print("\nRunning cross-validation...")
    results = cv.run_cross_validation()
    
    print("\nTest complete. Results saved to 'mini_cv_results' directory.")
    
    # Print summary of aggregate metrics
    if results:
        agg_metrics = results[0]["aggregate_metrics"]
        print("\nAggregate Metrics:")
        print(f"Value Accuracy: {agg_metrics['value_accuracy']['mean']:.2f}% ± {agg_metrics['value_accuracy']['std']:.2f}%")
        print(f"Success Rate: {agg_metrics['success_rate']['mean']:.2f}%")

if __name__ == "__main__":
    main()