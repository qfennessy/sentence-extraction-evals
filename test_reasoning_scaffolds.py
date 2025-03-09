#!/usr/bin/env python3
"""
Test whether adding reasoning scaffolds (chain-of-thought) improves extraction results.

This script compares a standard extraction prompt with a modified version
that includes step-by-step reasoning scaffolds, using the cross-validation
framework to evaluate performance differences.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from cross_validation import CrossValidator

def run_test():
    """
    Compare standard extraction prompt with a chain-of-thought version.
    """
    print("Testing if Chain-of-Thought reasoning scaffolds improve extraction results...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"cot_test_results/results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure cross-validation with both datasets
    cv = CrossValidator(
        datasets=["very-simple-family-sentences-evals.json"],
        models=["sonnet"],  # Using only Claude 3.7 Sonnet for consistency
        folds=3,            # 3-fold cross-validation 
        output_dir=output_dir
    )
    
    # Add prompt variations
    cv.add_prompt_variation("standard", "prompt_template.txt", "Standard Extraction")
    cv.add_prompt_variation("cot", "prompt_template_cot.txt", "Chain-of-Thought")
    
    # Run cross-validation
    print("\nRunning cross-validation...")
    results = cv.run_cross_validation()
    
    # Generate analysis report
    generate_report(results, output_dir)
    
    print(f"\nTest complete. Results saved to '{output_dir}'")

def generate_report(results, output_dir):
    """Generate comprehensive report comparing the two prompts."""
    # Group results by prompt
    standard_results = [r for r in results if r["prompt"] == "standard"]
    cot_results = [r for r in results if r["prompt"] == "cot"]
    
    # Calculate averages
    if standard_results and cot_results:
        standard_accuracy = sum(r["aggregate_metrics"]["value_accuracy"]["mean"] for r in standard_results) / len(standard_results)
        cot_accuracy = sum(r["aggregate_metrics"]["value_accuracy"]["mean"] for r in cot_results) / len(cot_results)
        
        standard_field_rate = sum(r["aggregate_metrics"]["field_extraction_rate"]["mean"] for r in standard_results) / len(standard_results)
        cot_field_rate = sum(r["aggregate_metrics"]["field_extraction_rate"]["mean"] for r in cot_results) / len(cot_results)
        
        standard_avg_score = sum(r["aggregate_metrics"]["avg_score"]["mean"] for r in standard_results) / len(standard_results)
        cot_avg_score = sum(r["aggregate_metrics"]["avg_score"]["mean"] for r in cot_results) / len(cot_results)
        
        # Create comparison table
        comparison = pd.DataFrame({
            "Metric": ["Value Accuracy", "Field Extraction Rate", "Average Score"],
            "Standard": [standard_accuracy, standard_field_rate, standard_avg_score],
            "Chain-of-Thought": [cot_accuracy, cot_field_rate, cot_avg_score],
            "Difference": [
                cot_accuracy - standard_accuracy,
                cot_field_rate - standard_field_rate,
                cot_avg_score - standard_avg_score
            ],
            "% Improvement": [
                (cot_accuracy - standard_accuracy) / standard_accuracy * 100 if standard_accuracy > 0 else 0,
                (cot_field_rate - standard_field_rate) / standard_field_rate * 100 if standard_field_rate > 0 else 0,
                (cot_avg_score - standard_avg_score) / standard_avg_score * 100 if standard_avg_score > 0 else 0
            ]
        })
        
        # Save comparison table
        comparison.to_csv(f"{output_dir}/summary.csv", index=False)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        metrics = ["Value Accuracy", "Field Extraction Rate", "Average Score"]
        standard_values = [standard_accuracy, standard_field_rate, standard_avg_score]
        cot_values = [cot_accuracy, cot_field_rate, cot_avg_score]
        
        x = range(len(metrics))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], standard_values, width, label='Standard')
        plt.bar([i + width/2 for i in x], cot_values, width, label='Chain-of-Thought')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Comparison of Standard vs Chain-of-Thought Prompts')
        plt.xticks(x, metrics)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(standard_values):
            plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        for i, v in enumerate(cot_values):
            plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.savefig(f"{output_dir}/comparison.png", dpi=300, bbox_inches='tight')
        
        # Create detailed markdown report
        with open(f"{output_dir}/reasoning_scaffolds_report.md", "w") as f:
            f.write("# Reasoning Scaffolds Evaluation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write("## Experiment Design\n\n")
            f.write("This experiment tests whether adding explicit reasoning scaffolds (chain-of-thought prompting) ")
            f.write("improves the accuracy of structured information extraction from natural language sentences.\n\n")
            
            f.write("### Methodology\n\n")
            f.write("- **Standard Prompt**: The baseline extraction prompt without reasoning scaffolds\n")
            f.write("- **Chain-of-Thought Prompt**: Added explicit step-by-step reasoning process\n")
            f.write("  1. Step 1: Identify all people mentioned in the sentence\n")
            f.write("  2. Step 2: Determine each person's relationships\n")
            f.write("  3. Step 3: Extract all biographical details for each person\n")
            f.write("  4. Step 4: Organize everything into the required JSON structure\n\n")
            
            f.write("### Evaluation Framework\n\n")
            f.write("- Cross-validation with 3 folds\n")
            f.write("- Test dataset: very-simple-family-sentences-evals.json\n")
            f.write("- Model tested: Claude 3.7 Sonnet\n\n")
            
            f.write("## Results Summary\n\n")
            
            # Format as markdown table
            f.write("| Metric | Standard | Chain-of-Thought | Difference | % Improvement |\n")
            f.write("|--------|----------|-----------------|------------|---------------|\n")
            for _, row in comparison.iterrows():
                f.write(f"| {row['Metric']} | {row['Standard']:.4f} | {row['Chain-of-Thought']:.4f} | {row['Difference']:.4f} | {row['% Improvement']:.2f}% |\n")
            
            f.write("\n\n![Comparison Chart](comparison.png)\n\n")
            
            # Analysis and discussion
            f.write("## Analysis\n\n")
            
            if cot_accuracy > standard_accuracy:
                f.write("### Value Accuracy\n\n")
                f.write(f"The Chain-of-Thought prompt improved value accuracy by {cot_accuracy - standard_accuracy:.4f} ")
                f.write(f"({(cot_accuracy - standard_accuracy) / standard_accuracy * 100:.2f}%). ")
                f.write("This suggests that guiding the model through an explicit reasoning process helps it extract more accurate values.\n\n")
            else:
                f.write("### Value Accuracy\n\n")
                f.write(f"The Chain-of-Thought prompt did not improve value accuracy, with a difference of {cot_accuracy - standard_accuracy:.4f} ")
                f.write(f"({(cot_accuracy - standard_accuracy) / standard_accuracy * 100:.2f}%). ")
                f.write("This suggests that for this specific task, explicit reasoning scaffolds may not improve extraction accuracy.\n\n")
            
            if cot_field_rate > standard_field_rate:
                f.write("### Field Extraction Rate\n\n")
                f.write(f"The Chain-of-Thought prompt improved field extraction rate by {cot_field_rate - standard_field_rate:.4f} ")
                f.write(f"({(cot_field_rate - standard_field_rate) / standard_field_rate * 100:.2f}%). ")
                f.write("This indicates that the reasoning scaffold helps the model identify more fields that should be extracted.\n\n")
            else:
                f.write("### Field Extraction Rate\n\n")
                f.write(f"The Chain-of-Thought prompt did not improve field extraction rate, with a difference of {cot_field_rate - standard_field_rate:.4f} ")
                f.write(f"({(cot_field_rate - standard_field_rate) / standard_field_rate * 100:.2f}%). ")
                f.write("This suggests that the explicit reasoning process did not help identify more fields to extract.\n\n")
            
            if cot_avg_score > standard_avg_score:
                f.write("### Average Score\n\n")
                f.write(f"The Chain-of-Thought prompt improved average sentence scores by {cot_avg_score - standard_avg_score:.4f} ")
                f.write(f"({(cot_avg_score - standard_avg_score) / standard_avg_score * 100:.2f}%). ")
                f.write("This combines both coverage and accuracy, suggesting an overall improvement in extraction quality.\n\n")
            else:
                f.write("### Average Score\n\n")
                f.write(f"The Chain-of-Thought prompt did not improve average sentence scores, with a difference of {cot_avg_score - standard_avg_score:.4f} ")
                f.write(f"({(cot_avg_score - standard_avg_score) / standard_avg_score * 100:.2f}%). ")
                f.write("This suggests that for overall extraction quality, explicit reasoning scaffolds did not provide benefits.\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            if cot_accuracy > standard_accuracy and cot_field_rate > standard_field_rate and cot_avg_score > standard_avg_score:
                f.write("The results show that adding reasoning scaffolds (chain-of-thought prompting) **improves** extraction performance ")
                f.write("across all measured metrics. The explicit step-by-step reasoning process helps the model better identify, ")
                f.write("analyze, and structure the information from natural language sentences.\n\n")
            elif cot_accuracy > standard_accuracy or cot_field_rate > standard_field_rate or cot_avg_score > standard_avg_score:
                f.write("The results show **mixed effects** from adding reasoning scaffolds. While some metrics showed improvement, ")
                f.write("others did not demonstrate clear benefits. This suggests that the effectiveness of reasoning scaffolds may ")
                f.write("depend on the specific aspects of the extraction task.\n\n")
            else:
                f.write("The results suggest that adding reasoning scaffolds (chain-of-thought prompting) **does not improve** extraction performance ")
                f.write("for this particular task. This could indicate that the model already performs well at structured extraction with clear ")
                f.write("instructions, and the additional reasoning steps may not provide significant benefits.\n\n")
            
            f.write("### Future Work\n\n")
            f.write("- Test on more complex datasets with nested relationships and ambiguous references\n")
            f.write("- Experiment with different reasoning scaffold designs and levels of detail\n")
            f.write("- Compare effects across different model sizes (Opus, Haiku) and model families\n")
            f.write("- Analyze specific cases where reasoning scaffolds help or hinder extraction\n")

if __name__ == "__main__":
    run_test()