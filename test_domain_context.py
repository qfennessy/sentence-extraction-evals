#!/usr/bin/env python3
"""
Test whether Claude-3.7 models benefit from domain context before instructions.

This script runs a cross-validation test to compare prompt variations:
1. With domain context paragraph before instructions
2. Without domain context (instructions only)

The comparison is run on the same dataset with Claude models to determine
if adding domain context before instructions improves extraction accuracy.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from cross_validation import CrossValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test whether Claude benefits from domain context before instructions")
    parser.add_argument("--dataset", default="very-simple-family-sentences-evals.json",
                      help="Dataset file to use for evaluation")
    parser.add_argument("--models", nargs="+", default=["sonnet"],
                      help="Claude models to test (default: sonnet)")
    parser.add_argument("--output-dir", default="domain_context_results",
                      help="Directory to store results")
    parser.add_argument("--folds", type=int, default=3,
                      help="Number of cross-validation folds")
    parser.add_argument("--max-sentences", type=int, default=10,
                      help="Maximum sentences to evaluate per fold")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = f"{args.output_dir}_{timestamp}"
    
    logger.info(f"Starting domain context test with configuration:")
    logger.info(f"- Dataset: {args.dataset}")
    logger.info(f"- Models: {args.models}")
    logger.info(f"- Output directory: {results_dir}")
    logger.info(f"- Folds: {args.folds}")
    logger.info(f"- Max sentences: {args.max_sentences}")
    
    # Check if files exist
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    if not os.path.exists("prompt_with_context.txt"):
        logger.error("prompt_with_context.txt not found")
        sys.exit(1)
        
    if not os.path.exists("prompt_without_context.txt"):
        logger.error("prompt_without_context.txt not found")
        sys.exit(1)
    
    logger.info("All required files found. Setting up cross-validator...")
    
    # Setup cross-validator
    try:
        cv = CrossValidator(
            datasets=[args.dataset],
            models=args.models,
            folds=args.folds,
            max_sentences=args.max_sentences,
            output_dir=results_dir,
        )
        
        # Add prompt variations
        logger.info("Adding prompt variations...")
        cv.add_prompt_variation(
            name="with_context",
            path="prompt_with_context.txt",
            description="Domain Context + Instructions"
        )
        
        cv.add_prompt_variation(
            name="without_context",
            path="prompt_without_context.txt",
            description="Instructions Only"
        )
        
        # Run cross-validation
        logger.info(f"Running cross-validation...")
        cv.run_cross_validation()
        
        logger.info("Test completed successfully!")
        logger.info(f"Check results at: {results_dir}/cv_results_*/")
    except Exception as e:
        logger.error(f"Error during test execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()