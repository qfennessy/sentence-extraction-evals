#!/usr/bin/env python3
"""
Cross-validation framework for extraction evaluation.

This module provides tools to systematically evaluate prompt variations
across different models and datasets, enabling rigorous comparison of
extraction performance.
"""

import os
import json
import subprocess
import pandas as pd
import numpy as np
import datetime
import itertools
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Define model groups for evaluation
MODEL_GROUPS = {
    "claude": ["sonnet", "opus", "haiku"],
    "openai": ["gpt4o", "gpt4", "gpt35"],
    "gemini": ["pro", "ultra"],
    "deepseek": ["coder", "chat"],
    "all_small": ["haiku", "gpt35", "pro", "chat"],
    "all_medium": ["sonnet", "gpt4", "pro"],
    "all_large": ["opus", "gpt4o", "ultra", "coder"],
}

class CrossValidator:
    """Framework for cross-validating extraction performance."""
    
    def __init__(
        self,
        eval_script_path: str = "claude_extraction_eval.py",
        output_dir: str = "cross_validation_results",
        base_prompt_path: str = "prompt_template.txt",
        datasets: List[str] = None,
        models: List[str] = None,
        model_group: str = None,
        folds: int = 5,
        max_sentences: int = None,
        temperature: float = 0.0,
    ):
        """Initialize the cross-validator.
        
        Args:
            eval_script_path: Path to the evaluation script
            output_dir: Directory to store results
            base_prompt_path: Path to the base prompt template
            datasets: List of dataset paths to evaluate
            models: List of specific models to evaluate
            model_group: Predefined group of models (overrides models if set)
            folds: Number of folds for cross-validation
            max_sentences: Maximum sentences to evaluate per dataset
            temperature: Temperature for model responses
        """
        self.eval_script_path = eval_script_path
        self.output_dir = Path(output_dir)
        self.base_prompt_path = base_prompt_path
        self.datasets = datasets or []
        self.prompt_variations = []
        
        # Set models based on group or direct list
        if model_group and model_group in MODEL_GROUPS:
            self.models = MODEL_GROUPS[model_group]
        else:
            self.models = models or ["sonnet"]
            
        self.folds = folds
        self.max_sentences = max_sentences
        self.temperature = temperature
        self.results = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def add_prompt_variation(self, name: str, path: str, description: str = None):
        """Add a prompt variation to evaluate.
        
        Args:
            name: Short name for the prompt variation
            path: Path to the prompt template file
            description: Longer description of the prompt variation
        """
        self.prompt_variations.append({
            "name": name,
            "path": path,
            "description": description or name
        })
        
    def prepare_dataset_folds(self, dataset_path: str) -> List[str]:
        """Split a dataset into folds for cross-validation.
        
        Args:
            dataset_path: Path to the dataset file
            
        Returns:
            List of paths to the fold dataset files
        """
        # Create a directory for the folds
        dataset_name = Path(dataset_path).stem
        folds_dir = self.output_dir / f"{dataset_name}_folds"
        os.makedirs(folds_dir, exist_ok=True)
        
        # Load the dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            
        sentences = data.get("sentences", [])
        total_sentences = len(sentences)
        
        # If too few sentences, adjust folds
        actual_folds = min(self.folds, total_sentences)
        if actual_folds < self.folds:
            print(f"Warning: Dataset {dataset_path} has only {total_sentences} sentences. "
                  f"Reducing folds to {actual_folds}.")
        
        # Create fold files
        fold_paths = []
        for i in range(actual_folds):
            # Select fold sentences using modulo to distribute evenly
            fold_sentences = [s for idx, s in enumerate(sentences) if idx % actual_folds == i]
            
            fold_data = {"sentences": fold_sentences}
            fold_path = folds_dir / f"fold_{i+1}.json"
            
            with open(fold_path, 'w') as f:
                json.dump(fold_data, f, indent=2)
                
            fold_paths.append(str(fold_path))
            
        return fold_paths
    
    def run_evaluation(self, prompt_path: str, model: str, dataset_path: str) -> Dict[str, Any]:
        """Run a single evaluation using the evaluation script.
        
        Args:
            prompt_path: Path to the prompt template
            model: Model identifier to use
            dataset_path: Path to the dataset
            
        Returns:
            Dictionary with evaluation results
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{Path(dataset_path).stem}_{model}_{Path(prompt_path).stem}_{timestamp}"
        
        # Prepare command
        cmd = [
            "python", self.eval_script_path,
            "--model", model,
            "--prompt-file", prompt_path,
            "--eval-data", dataset_path,
            "--temperature", str(self.temperature)
        ]
        
        if self.max_sentences:
            cmd.extend(["--max-sentences", str(self.max_sentences)])
        
        # Run evaluation
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse output to extract key metrics
            output_lines = result.stdout.split("\n")
            
            # Extract metrics from output
            metrics = {}
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
            
            # Find the output directory from stdout
            output_dir = None
            for line in output_lines:
                if "Output directory:" in line:
                    output_dir = line.split("Output directory:")[1].strip()
                    break
            
            # Read full results from output files if available
            detailed_results = {}
            if output_dir and os.path.exists(output_dir):
                # Read summary CSV
                summary_path = os.path.join(output_dir, "summary.csv")
                if os.path.exists(summary_path):
                    summary_df = pd.read_csv(summary_path)
                    detailed_results["summary"] = summary_df.to_dict(orient="records")
                
                # Read field accuracy CSV
                field_path = os.path.join(output_dir, "field_accuracy.csv")
                if os.path.exists(field_path):
                    field_df = pd.read_csv(field_path)
                    detailed_results["field_accuracy"] = field_df.to_dict(orient="records")
            
            return {
                "run_id": run_id,
                "model": model,
                "prompt": Path(prompt_path).stem,
                "dataset": Path(dataset_path).stem,
                "metrics": metrics,
                "detailed_results": detailed_results,
                "success": True
            }
            
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return {
                "run_id": run_id,
                "model": model,
                "prompt": Path(prompt_path).stem,
                "dataset": Path(dataset_path).stem,
                "metrics": {},
                "success": False,
                "error": str(e)
            }
    
    def run_cross_validation(self):
        """Run cross-validation across all datasets, models, and prompt variations."""
        # Add base prompt if no variations added
        if not self.prompt_variations:
            self.add_prompt_variation("base", self.base_prompt_path, "Base Prompt")
        
        print(f"Running cross-validation with:")
        print(f"- {len(self.datasets)} datasets")
        print(f"- {len(self.models)} models: {', '.join(self.models)}")
        print(f"- {len(self.prompt_variations)} prompt variations")
        print(f"- {self.folds} folds per dataset")
        
        all_results = []
        
        # Loop through all combinations
        for dataset_path in self.datasets:
            dataset_name = Path(dataset_path).stem
            print(f"\nProcessing dataset: {dataset_name}")
            
            # Create folds for this dataset
            fold_paths = self.prepare_dataset_folds(dataset_path)
            
            for model in self.models:
                print(f"  Evaluating model: {model}")
                
                for prompt_var in self.prompt_variations:
                    prompt_name = prompt_var["name"]
                    prompt_path = prompt_var["path"]
                    
                    print(f"    Testing prompt: {prompt_name}")
                    
                    # Run evaluation on each fold
                    fold_results = []
                    for i, fold_path in enumerate(fold_paths):
                        print(f"      Fold {i+1}/{len(fold_paths)}...")
                        result = self.run_evaluation(prompt_path, model, fold_path)
                        result["fold"] = i + 1
                        fold_results.append(result)
                    
                    # Calculate aggregate metrics across folds
                    aggregate_metrics = self._aggregate_fold_results(fold_results)
                    
                    # Store full result with all folds and aggregate metrics
                    full_result = {
                        "dataset": dataset_name,
                        "model": model,
                        "prompt": prompt_name,
                        "prompt_description": prompt_var["description"],
                        "fold_results": fold_results,
                        "aggregate_metrics": aggregate_metrics
                    }
                    
                    all_results.append(full_result)
        
        self.results = all_results
        self._save_results()
        return all_results
    
    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across folds.
        
        Args:
            fold_results: Results from individual folds
            
        Returns:
            Dictionary with aggregated metrics
        """
        # Extract metrics from successful folds
        successful_folds = [r for r in fold_results if r["success"]]
        if not successful_folds:
            return {
                "value_accuracy": None,
                "field_extraction_rate": None,
                "avg_sentence_score": None,
                "success_rate": None,
                "successful_folds": 0,
                "total_folds": len(fold_results)
            }
        
        # Get list of each metric across folds
        value_accuracies = [r["metrics"].get("value_accuracy", 0) for r in successful_folds]
        field_rates = [r["metrics"].get("field_extraction_rate", 0) for r in successful_folds]
        avg_scores = [r["metrics"].get("avg_sentence_score", 0) for r in successful_folds]
        success_rates = [r["metrics"].get("success_rate", 0) for r in successful_folds]
        
        # Calculate aggregate statistics
        return {
            "value_accuracy": {
                "mean": np.mean(value_accuracies),
                "std": np.std(value_accuracies),
                "min": np.min(value_accuracies),
                "max": np.max(value_accuracies),
                "values": value_accuracies
            },
            "field_extraction_rate": {
                "mean": np.mean(field_rates),
                "std": np.std(field_rates),
                "min": np.min(field_rates),
                "max": np.max(field_rates),
                "values": field_rates
            },
            "avg_sentence_score": {
                "mean": np.mean(avg_scores),
                "std": np.std(avg_scores),
                "min": np.min(avg_scores),
                "max": np.max(avg_scores),
                "values": avg_scores
            },
            "success_rate": {
                "mean": np.mean(success_rates),
                "std": np.std(success_rates),
                "min": np.min(success_rates),
                "max": np.max(success_rates),
                "values": success_rates
            },
            "successful_folds": len(successful_folds),
            "total_folds": len(fold_results)
        }
    
    def _save_results(self):
        """Save the cross-validation results."""
        if not self.results:
            print("No results to save")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = self.output_dir / f"cv_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save full results as JSON
        with open(results_dir / "full_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary CSV
        summary_rows = []
        for result in self.results:
            if "aggregate_metrics" not in result:
                continue
                
            agg = result["aggregate_metrics"]
            row = {
                "dataset": result["dataset"],
                "model": result["model"],
                "prompt": result["prompt"],
                "prompt_description": result["prompt_description"],
                "value_accuracy_mean": agg["value_accuracy"]["mean"] if "value_accuracy" in agg else None,
                "value_accuracy_std": agg["value_accuracy"]["std"] if "value_accuracy" in agg else None,
                "field_extraction_rate_mean": agg["field_extraction_rate"]["mean"] if "field_extraction_rate" in agg else None,
                "avg_sentence_score_mean": agg["avg_sentence_score"]["mean"] if "avg_sentence_score" in agg else None,
                "success_rate_mean": agg["success_rate"]["mean"] if "success_rate" in agg else None,
                "successful_folds": agg["successful_folds"],
                "total_folds": agg["total_folds"]
            }
            summary_rows.append(row)
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(results_dir / "summary.csv", index=False)
            
            # Create visualization
            self._create_visualizations(summary_df, results_dir)
        
        print(f"Results saved to {results_dir}")
    
    def _create_visualizations(self, summary_df: pd.DataFrame, output_dir: Path):
        """Create visualizations for the cross-validation results.
        
        Args:
            summary_df: Summary dataframe with cross-validation results
            output_dir: Directory to save visualizations
        """
        # Set Seaborn style
        sns.set_theme(style="whitegrid")
        
        # 1. Value accuracy comparison across models and prompts
        plt.figure(figsize=(12, 8))
        chart = sns.barplot(
            data=summary_df, 
            x="model", 
            y="value_accuracy_mean", 
            hue="prompt",
            errorbar=("ci", 95)
        )
        chart.set_title("Value Accuracy by Model and Prompt")
        chart.set_xlabel("Model")
        chart.set_ylabel("Value Accuracy (%)")
        plt.tight_layout()
        plt.savefig(output_dir / "value_accuracy_by_model_prompt.png")
        
        # 2. Value accuracy comparison across datasets
        if len(summary_df["dataset"].unique()) > 1:
            plt.figure(figsize=(12, 8))
            chart = sns.barplot(
                data=summary_df, 
                x="dataset", 
                y="value_accuracy_mean", 
                hue="prompt",
                errorbar=("ci", 95)
            )
            chart.set_title("Value Accuracy by Dataset and Prompt")
            chart.set_xlabel("Dataset")
            chart.set_ylabel("Value Accuracy (%)")
            plt.tight_layout()
            plt.savefig(output_dir / "value_accuracy_by_dataset_prompt.png")
        
        # 3. Heatmap of performance across models and prompts
        plt.figure(figsize=(10, 6))
        pivot_data = summary_df.pivot_table(
            index="model", 
            columns="prompt", 
            values="value_accuracy_mean",
            aggfunc="mean"
        )
        sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlGnBu")
        plt.title("Value Accuracy Heatmap by Model and Prompt")
        plt.tight_layout()
        plt.savefig(output_dir / "value_accuracy_heatmap.png")
        
        # 4. Create HTML report
        html_report = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cross-Validation Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .chart-container { margin: 20px 0; }
                .chart { width: 100%; max-width: 800px; }
            </style>
        </head>
        <body>
            <h1>Cross-Validation Results</h1>
        """
        
        # Add summary table
        html_report += "<h2>Summary</h2>"
        html_report += summary_df.to_html(index=False)
        
        # Add charts
        html_report += """
            <h2>Visualizations</h2>
            <div class="chart-container">
                <h3>Value Accuracy by Model and Prompt</h3>
                <img class="chart" src="value_accuracy_by_model_prompt.png" alt="Value Accuracy Chart">
            </div>
        """
        
        if len(summary_df["dataset"].unique()) > 1:
            html_report += """
                <div class="chart-container">
                    <h3>Value Accuracy by Dataset and Prompt</h3>
                    <img class="chart" src="value_accuracy_by_dataset_prompt.png" alt="Value Accuracy by Dataset">
                </div>
            """
        
        html_report += """
            <div class="chart-container">
                <h3>Value Accuracy Heatmap</h3>
                <img class="chart" src="value_accuracy_heatmap.png" alt="Value Accuracy Heatmap">
            </div>
        </body>
        </html>
        """
        
        with open(output_dir / "report.html", 'w') as f:
            f.write(html_report)

def main():
    """Main function to run the cross-validation script."""
    parser = argparse.ArgumentParser(description="Cross-validate extraction models and prompts")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset files to evaluate")
    parser.add_argument("--prompt-variations", nargs="+", help="Prompt template files to evaluate")
    parser.add_argument("--model-group", choices=list(MODEL_GROUPS.keys()), 
                        help="Predefined group of models to evaluate")
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--max-sentences", type=int, help="Maximum sentences to evaluate per fold")
    parser.add_argument("--output-dir", default="cross_validation_results", 
                        help="Directory to store results")
    parser.add_argument("--base-prompt", default="prompt_template.txt", 
                        help="Base prompt template path")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_group and not args.models:
        parser.error("Either --model-group or --models must be specified")
    
    # Convert prompt variations to list of dicts
    prompt_vars = []
    if args.prompt_variations:
        for i, path in enumerate(args.prompt_variations):
            name = f"var_{i+1}" if i > 0 else "base"
            prompt_vars.append({
                "name": name,
                "path": path,
                "description": f"Prompt Variation {i+1}"
            })
    
    # Initialize and run cross-validator
    cv = CrossValidator(
        datasets=args.datasets,
        models=args.models,
        model_group=args.model_group,
        folds=args.folds,
        max_sentences=args.max_sentences,
        output_dir=args.output_dir,
        base_prompt_path=args.base_prompt
    )
    
    # Add prompt variations if specified
    for var in prompt_vars:
        cv.add_prompt_variation(var["name"], var["path"], var["description"])
    
    # Run cross-validation
    cv.run_cross_validation()

if __name__ == "__main__":
    main()