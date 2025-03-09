#!/usr/bin/env python3
"""
Run a comparative test of the cross-validation framework.
Tests different model performance on datasets of varying complexity.
"""

from cross_validation import CrossValidator

def main():
    """Run comparative cross-validation."""
    print("Running comparative cross-validation test...")
    
    # Test with two dataset complexity levels and two models
    # Using limited sentences to speed up the test
    cv = CrossValidator(
        datasets=[
            "very-simple-family-sentences-evals.json",  # Simple dataset
            "test-subset-2.json"                        # More complex dataset
        ],
        models=["sonnet", "haiku"],  # Different model sizes
        folds=2,
        max_sentences=3,
        output_dir="comparative_cv_results"
    )
    
    # Use the same prompt for all tests
    cv.add_prompt_variation("current", "prompt_template.txt", "Current Prompt Template")
    
    # Run full cross-validation
    print("\nRunning full cross-validation...")
    results = cv.run_cross_validation()
    
    print("\nTest complete. Results saved to 'comparative_cv_results' directory.")

if __name__ == "__main__":
    main()