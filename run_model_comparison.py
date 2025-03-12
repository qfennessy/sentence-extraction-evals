#\!/usr/bin/env python3
"""
Script to compare Claude 3.7 and GPT-4o using the optimized prompt.
"""

import subprocess
import time
import os
from pathlib import Path

def main():
    # Create output directory
    output_dir = Path("comparative_cv_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define the models to test
    models = [
        "sonnet",  # Claude 3.7
        "gpt4o"    # GPT-4o
    ]
    
    # Run evaluations
    for model in models:
        print(f"\n=== Testing {model} ===\n")
        
        cmd = [
            "python", "claude_extraction_eval.py",
            "--model", model,
            "--prompt-file", "prompt_optimized.txt",
            "--eval-data", "family-sentences-evals.json",
            "--max-sentences", "10"  # Limit to 10 sentences for quicker testing
        ]
        
        # Run the command
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation for {model}: {e}")
    
    print("\nBoth evaluations complete. Check the eval-output directory for detailed results.")

if __name__ == "__main__":
    main()
