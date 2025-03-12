#!/usr/bin/env python
"""
Prompt Tracker: A tool for tracking prompt enhancements and test results.

This tool provides a centralized system for logging prompt versions,
their test results, and performance metrics over time, allowing for
systematic evaluation of prompt effectiveness.
"""

import json
import os
import shutil
import datetime
import hashlib
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# Constants
TRACKER_DIR = "prompt-tracking"
PROMPTS_DIR = os.path.join(TRACKER_DIR, "prompts")
EVALUATIONS_DIR = os.path.join(TRACKER_DIR, "evaluations")
REGISTRY_FILE = os.path.join(TRACKER_DIR, "prompt_registry.json")
PERFORMANCE_FILE = os.path.join(TRACKER_DIR, "performance_history.json")
REFINEMENT_FILE = os.path.join(TRACKER_DIR, "refinement_lineage.json")
PROMPT_PREFIX = "prompt_"

class PromptTracker:
    """
    A system for tracking prompt enhancements and their test results.
    """
    
    def __init__(self):
        """Initialize the prompt tracker system."""
        self._ensure_directories()
        self._load_registry()
        self._load_performance()
        self._load_refinements()
        
    def _ensure_directories(self):
        """Ensure all necessary directories exist."""
        os.makedirs(PROMPTS_DIR, exist_ok=True)
        os.makedirs(EVALUATIONS_DIR, exist_ok=True)
        
        # Initialize registry files if they don't exist
        for file_path in [REGISTRY_FILE, PERFORMANCE_FILE, REFINEMENT_FILE]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump({}, f, indent=2)
    
    def _load_registry(self):
        """Load the prompt registry."""
        with open(REGISTRY_FILE, 'r') as f:
            self.registry = json.load(f)
    
    def _load_performance(self):
        """Load the performance history."""
        with open(PERFORMANCE_FILE, 'r') as f:
            self.performance = json.load(f)
    
    def _load_refinements(self):
        """Load the refinement lineage."""
        with open(REFINEMENT_FILE, 'r') as f:
            self.refinements = json.load(f)
            
    def _save_registry(self):
        """Save the prompt registry."""
        with open(REGISTRY_FILE, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _save_performance(self):
        """Save the performance history."""
        with open(PERFORMANCE_FILE, 'w') as f:
            json.dump(self.performance, f, indent=2)
    
    def _save_refinements(self):
        """Save the refinement lineage."""
        with open(REFINEMENT_FILE, 'w') as f:
            json.dump(self.refinements, f, indent=2)
    
    def _generate_prompt_id(self, prompt_content):
        """Generate a unique ID for a prompt based on its content."""
        # Use a shortened hash of the content for the ID
        prompt_hash = hashlib.md5(prompt_content.encode('utf-8')).hexdigest()[:8]
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        return f"{PROMPT_PREFIX}{timestamp}_{prompt_hash}"
    
    def register_prompt(self, prompt_file, name=None, description=None, parent_id=None, tags=None):
        """
        Register a new prompt or a refinement of an existing prompt.
        
        Args:
            prompt_file: Path to the prompt file
            name: Optional friendly name for the prompt
            description: Optional description of the prompt
            parent_id: ID of the parent prompt if this is a refinement
            tags: Optional list of tags for categorizing prompts
        
        Returns:
            prompt_id: The unique ID assigned to the prompt
        """
        # Read the prompt content
        with open(prompt_file, 'r') as f:
            prompt_content = f.read()
        
        # Generate a unique ID
        prompt_id = self._generate_prompt_id(prompt_content)
        
        # Check if this prompt already exists
        for existing_id, details in self.registry.items():
            if details.get("hash") == hashlib.md5(prompt_content.encode('utf-8')).hexdigest():
                print(f"This prompt already exists with ID: {existing_id}")
                return existing_id
        
        # Prepare timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Copy the prompt file to our tracking directory
        prompt_filename = f"{prompt_id}.txt"
        prompt_path = os.path.join(PROMPTS_DIR, prompt_filename)
        shutil.copy(prompt_file, prompt_path)
        
        # Add to registry
        self.registry[prompt_id] = {
            "name": name or os.path.basename(prompt_file),
            "description": description or "",
            "timestamp": timestamp,
            "file_path": prompt_path,
            "hash": hashlib.md5(prompt_content.encode('utf-8')).hexdigest(),
            "tags": tags or [],
            "performance_summary": {
                "best_score": 0,
                "latest_score": 0,
                "evaluation_count": 0
            }
        }
        
        # If this is a refinement, add to refinement lineage
        if parent_id:
            if parent_id not in self.refinements:
                self.refinements[parent_id] = {"children": []}
            
            self.refinements[parent_id]["children"].append({
                "prompt_id": prompt_id,
                "timestamp": timestamp,
                "description": description or ""
            })
        
        # Save registry and refinements
        self._save_registry()
        self._save_refinements()
        
        print(f"Prompt registered with ID: {prompt_id}")
        return prompt_id
    
    def log_evaluation(self, prompt_id, results_file, model_name, dataset_name, metrics=None, notes=None):
        """
        Log the results of evaluating a prompt.
        
        Args:
            prompt_id: ID of the prompt that was evaluated
            results_file: Path to the evaluation results file
            model_name: Name of the model used
            dataset_name: Name of the dataset used
            metrics: Dictionary of performance metrics
            notes: Optional notes about the evaluation
            
        Returns:
            evaluation_id: The unique ID assigned to this evaluation
        """
        if prompt_id not in self.registry:
            print(f"Error: Prompt ID {prompt_id} not found in registry.")
            return None
        
        timestamp = datetime.datetime.now()
        timestamp_str = timestamp.isoformat()
        
        # Generate evaluation ID
        evaluation_id = f"eval_{prompt_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create evaluation directory
        eval_dir = os.path.join(EVALUATIONS_DIR, evaluation_id)
        os.makedirs(eval_dir, exist_ok=True)
        
        # Copy results file to evaluation directory
        results_filename = os.path.basename(results_file)
        results_path = os.path.join(eval_dir, results_filename)
        shutil.copy(results_file, results_path)
        
        # Copy prompt file to evaluation directory for reference
        prompt_file = self.registry[prompt_id]["file_path"]
        prompt_path = os.path.join(eval_dir, "prompt.txt")
        shutil.copy(prompt_file, prompt_path)
        
        # Parse metrics from results file if not provided
        if not metrics and results_file.endswith('.json'):
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                
                # Try to extract metrics from common formats
                if 'metrics' in results_data:
                    metrics = results_data['metrics']
                elif 'summary' in results_data:
                    metrics = results_data['summary']
        
        # Create metadata file
        metadata = {
            "prompt_id": prompt_id,
            "model": model_name,
            "dataset": dataset_name,
            "timestamp": timestamp_str,
            "results_file": results_path,
            "metrics": metrics or {},
            "notes": notes or ""
        }
        
        metadata_path = os.path.join(eval_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update performance history
        if prompt_id not in self.performance:
            self.performance[prompt_id] = []
        
        performance_entry = {
            "evaluation_id": evaluation_id,
            "model": model_name,
            "dataset": dataset_name,
            "timestamp": timestamp_str,
            "metrics": metrics or {},
            "notes": notes or ""
        }
        
        self.performance[prompt_id].append(performance_entry)
        
        # Update registry with performance summary
        if metrics and 'overall_accuracy' in metrics:
            accuracy = metrics['overall_accuracy']
            
            # Update best score
            current_best = self.registry[prompt_id]["performance_summary"]["best_score"]
            if accuracy > current_best:
                self.registry[prompt_id]["performance_summary"]["best_score"] = accuracy
            
            # Update latest score
            self.registry[prompt_id]["performance_summary"]["latest_score"] = accuracy
        
        # Increment evaluation count
        self.registry[prompt_id]["performance_summary"]["evaluation_count"] += 1
        
        # Save updates
        self._save_performance()
        self._save_registry()
        
        print(f"Evaluation logged with ID: {evaluation_id}")
        return evaluation_id
    
    def get_prompt_performance(self, prompt_id):
        """
        Get the performance history for a specific prompt.
        
        Args:
            prompt_id: ID of the prompt
            
        Returns:
            List of evaluation entries for the prompt
        """
        if prompt_id not in self.performance:
            print(f"No performance data found for prompt ID: {prompt_id}")
            return []
        
        return self.performance[prompt_id]
    
    def list_prompts(self, tag=None, min_accuracy=None, limit=None):
        """
        List all prompts in the registry, optionally filtered.
        
        Args:
            tag: Optional tag to filter by
            min_accuracy: Optional minimum accuracy to filter by
            limit: Optional maximum number of prompts to return
            
        Returns:
            List of prompt entries
        """
        results = []
        
        for prompt_id, details in self.registry.items():
            # Apply tag filter if specified
            if tag and tag not in details["tags"]:
                continue
            
            # Apply accuracy filter if specified
            if min_accuracy is not None:
                best_score = details["performance_summary"]["best_score"]
                if best_score < min_accuracy:
                    continue
            
            results.append({
                "prompt_id": prompt_id,
                "name": details["name"],
                "timestamp": details["timestamp"],
                "best_score": details["performance_summary"]["best_score"],
                "latest_score": details["performance_summary"]["latest_score"],
                "eval_count": details["performance_summary"]["evaluation_count"],
                "tags": details["tags"]
            })
        
        # Sort by best score (descending)
        results.sort(key=lambda x: x["best_score"], reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            results = results[:limit]
        
        return results
    
    def list_evaluations(self, prompt_id=None, model=None, dataset=None, limit=None):
        """
        List evaluations, optionally filtered.
        
        Args:
            prompt_id: Optional prompt ID to filter by
            model: Optional model name to filter by
            dataset: Optional dataset name to filter by
            limit: Optional maximum number of evaluations to return
            
        Returns:
            List of evaluation entries
        """
        results = []
        
        for pid, evaluations in self.performance.items():
            # Skip if filtering by prompt_id and this isn't it
            if prompt_id and pid != prompt_id:
                continue
            
            for eval_entry in evaluations:
                # Apply model filter if specified
                if model and eval_entry["model"] != model:
                    continue
                
                # Apply dataset filter if specified
                if dataset and eval_entry["dataset"] != dataset:
                    continue
                
                results.append({
                    "evaluation_id": eval_entry["evaluation_id"],
                    "prompt_id": pid,
                    "model": eval_entry["model"],
                    "dataset": eval_entry["dataset"],
                    "timestamp": eval_entry["timestamp"],
                    "metrics": eval_entry["metrics"]
                })
        
        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Apply limit if specified
        if limit is not None:
            results = results[:limit]
        
        return results
    
    def get_best_prompts(self, model=None, dataset=None, limit=5):
        """
        Get the best performing prompts, optionally filtered by model and dataset.
        
        Args:
            model: Optional model name to filter by
            dataset: Optional dataset name to filter by
            limit: Maximum number of prompts to return
            
        Returns:
            List of best performing prompts
        """
        # Collect all evaluations that match the filters
        filtered_evals = []
        
        for prompt_id, evaluations in self.performance.items():
            for eval_entry in evaluations:
                # Apply model filter if specified
                if model and eval_entry["model"] != model:
                    continue
                
                # Apply dataset filter if specified
                if dataset and eval_entry["dataset"] != dataset:
                    continue
                
                # Skip entries without metrics or overall_accuracy
                if not eval_entry.get("metrics") or "overall_accuracy" not in eval_entry["metrics"]:
                    continue
                
                filtered_evals.append({
                    "prompt_id": prompt_id,
                    "model": eval_entry["model"],
                    "dataset": eval_entry["dataset"],
                    "timestamp": eval_entry["timestamp"],
                    "accuracy": eval_entry["metrics"]["overall_accuracy"]
                })
        
        # Group by prompt_id and get the best score for each
        best_by_prompt = {}
        for eval_entry in filtered_evals:
            prompt_id = eval_entry["prompt_id"]
            accuracy = eval_entry["accuracy"]
            
            if prompt_id not in best_by_prompt or accuracy > best_by_prompt[prompt_id]["accuracy"]:
                best_by_prompt[prompt_id] = eval_entry
        
        # Sort by accuracy (descending)
        results = list(best_by_prompt.values())
        results.sort(key=lambda x: x["accuracy"], reverse=True)
        
        # Apply limit
        results = results[:limit]
        
        # Add prompt details from registry
        for entry in results:
            prompt_id = entry["prompt_id"]
            if prompt_id in self.registry:
                entry["name"] = self.registry[prompt_id]["name"]
                entry["description"] = self.registry[prompt_id]["description"]
        
        return results
    
    def get_performance_trend(self, prompt_ids=None, model=None, dataset=None):
        """
        Generate performance trend data for specified prompts.
        
        Args:
            prompt_ids: List of prompt IDs to include (or None for all)
            model: Optional model name to filter by
            dataset: Optional dataset name to filter by
            
        Returns:
            DataFrame with performance trend data
        """
        trend_data = []
        
        # If no prompt_ids specified, use all
        if not prompt_ids:
            prompt_ids = list(self.performance.keys())
        
        for prompt_id in prompt_ids:
            if prompt_id not in self.performance:
                continue
            
            for eval_entry in self.performance[prompt_id]:
                # Apply model filter if specified
                if model and eval_entry["model"] != model:
                    continue
                
                # Apply dataset filter if specified
                if dataset and eval_entry["dataset"] != dataset:
                    continue
                
                # Skip entries without metrics or overall_accuracy
                if not eval_entry.get("metrics") or "overall_accuracy" not in eval_entry["metrics"]:
                    continue
                
                prompt_name = self.registry[prompt_id]["name"] if prompt_id in self.registry else prompt_id
                
                trend_data.append({
                    "prompt_id": prompt_id,
                    "prompt_name": prompt_name,
                    "timestamp": eval_entry["timestamp"],
                    "model": eval_entry["model"],
                    "dataset": eval_entry["dataset"],
                    "accuracy": eval_entry["metrics"]["overall_accuracy"]
                })
        
        # Convert to DataFrame
        if trend_data:
            df = pd.DataFrame(trend_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            return df
        
        return pd.DataFrame()
    
    def plot_performance_trend(self, prompt_ids=None, model=None, dataset=None, output_file=None):
        """
        Plot performance trend for specified prompts.
        
        Args:
            prompt_ids: List of prompt IDs to include (or None for all)
            model: Optional model name to filter by
            dataset: Optional dataset name to filter by
            output_file: Optional file path to save the plot
            
        Returns:
            None
        """
        df = self.get_performance_trend(prompt_ids, model, dataset)
        
        if df.empty:
            print("No performance data found for the specified filters.")
            return
        
        plt.figure(figsize=(12, 6))
        
        for prompt_name, group in df.groupby("prompt_name"):
            plt.plot(group["timestamp"], group["accuracy"], marker="o", label=prompt_name)
        
        plt.xlabel("Date")
        plt.ylabel("Accuracy")
        plt.title("Prompt Performance Trend")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save if output_file specified
        if output_file:
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
        
        plt.show()
    
    def generate_report(self, output_file=None):
        """
        Generate a comprehensive report of prompt performance.
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            Report text
        """
        report_lines = []
        
        # Title
        report_lines.append("# Prompt Performance Report")
        report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("\n")
        
        # Best performing prompts
        report_lines.append("## Best Performing Prompts")
        best_prompts = self.get_best_prompts(limit=5)
        
        if best_prompts:
            headers = ["Prompt ID", "Name", "Model", "Dataset", "Accuracy", "Date"]
            table_data = []
            
            for entry in best_prompts:
                table_data.append([
                    entry["prompt_id"],
                    entry["name"],
                    entry["model"],
                    entry["dataset"],
                    f"{entry['accuracy'] * 100:.2f}%",
                    entry["timestamp"].split("T")[0]
                ])
            
            report_lines.append(tabulate(table_data, headers=headers, tablefmt="pipe"))
        else:
            report_lines.append("No prompt evaluations found.")
        
        report_lines.append("\n")
        
        # Recent evaluations
        report_lines.append("## Recent Evaluations")
        recent_evals = self.list_evaluations(limit=10)
        
        if recent_evals:
            headers = ["Evaluation ID", "Prompt ID", "Model", "Dataset", "Accuracy", "Date"]
            table_data = []
            
            for entry in recent_evals:
                accuracy = entry["metrics"].get("overall_accuracy", "N/A")
                if isinstance(accuracy, (int, float)):
                    accuracy = f"{accuracy * 100:.2f}%"
                
                table_data.append([
                    entry["evaluation_id"],
                    entry["prompt_id"],
                    entry["model"],
                    entry["dataset"],
                    accuracy,
                    entry["timestamp"].split("T")[0]
                ])
            
            report_lines.append(tabulate(table_data, headers=headers, tablefmt="pipe"))
        else:
            report_lines.append("No recent evaluations found.")
        
        report_lines.append("\n")
        
        # Model performance comparison
        report_lines.append("## Model Performance Comparison")
        
        # Get all evaluations and group by model
        all_evals = self.list_evaluations()
        model_metrics = {}
        
        for entry in all_evals:
            if "metrics" not in entry or "overall_accuracy" not in entry["metrics"]:
                continue
            
            model = entry["model"]
            accuracy = entry["metrics"]["overall_accuracy"]
            
            if model not in model_metrics:
                model_metrics[model] = []
            
            model_metrics[model].append(accuracy)
        
        if model_metrics:
            headers = ["Model", "Avg Accuracy", "Max Accuracy", "Min Accuracy", "Evaluations"]
            table_data = []
            
            for model, accuracies in model_metrics.items():
                table_data.append([
                    model,
                    f"{sum(accuracies) / len(accuracies) * 100:.2f}%",
                    f"{max(accuracies) * 100:.2f}%",
                    f"{min(accuracies) * 100:.2f}%",
                    len(accuracies)
                ])
            
            # Sort by average accuracy
            table_data.sort(key=lambda x: float(x[1].rstrip('%')), reverse=True)
            
            report_lines.append(tabulate(table_data, headers=headers, tablefmt="pipe"))
        else:
            report_lines.append("No model performance data found.")
        
        # Combine report
        report_text = "\n".join(report_lines)
        
        # Save if output_file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Prompt Tracker: Track prompt enhancements and test results")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Register prompt command
    register_parser = subparsers.add_parser("register", help="Register a new prompt")
    register_parser.add_argument("prompt_file", help="Path to the prompt file")
    register_parser.add_argument("--name", help="Friendly name for the prompt")
    register_parser.add_argument("--description", help="Description of the prompt")
    register_parser.add_argument("--parent", help="ID of the parent prompt if this is a refinement")
    register_parser.add_argument("--tags", help="Comma-separated list of tags")
    
    # Log evaluation command
    log_parser = subparsers.add_parser("log", help="Log evaluation results")
    log_parser.add_argument("prompt_id", help="ID of the prompt that was evaluated")
    log_parser.add_argument("results_file", help="Path to the evaluation results file")
    log_parser.add_argument("--model", required=True, help="Name of the model used")
    log_parser.add_argument("--dataset", required=True, help="Name of the dataset used")
    log_parser.add_argument("--notes", help="Notes about the evaluation")
    
    # List prompts command
    list_prompts_parser = subparsers.add_parser("list-prompts", help="List registered prompts")
    list_prompts_parser.add_argument("--tag", help="Filter by tag")
    list_prompts_parser.add_argument("--min-accuracy", type=float, help="Filter by minimum accuracy")
    list_prompts_parser.add_argument("--limit", type=int, help="Maximum number of prompts to return")
    
    # List evaluations command
    list_evals_parser = subparsers.add_parser("list-evals", help="List evaluations")
    list_evals_parser.add_argument("--prompt", help="Filter by prompt ID")
    list_evals_parser.add_argument("--model", help="Filter by model name")
    list_evals_parser.add_argument("--dataset", help="Filter by dataset name")
    list_evals_parser.add_argument("--limit", type=int, help="Maximum number of evaluations to return")
    
    # Get best prompts command
    best_parser = subparsers.add_parser("best", help="Get best performing prompts")
    best_parser.add_argument("--model", help="Filter by model name")
    best_parser.add_argument("--dataset", help="Filter by dataset name")
    best_parser.add_argument("--limit", type=int, default=5, help="Maximum number of prompts to return")
    
    # Plot performance trend command
    plot_parser = subparsers.add_parser("plot", help="Plot performance trend")
    plot_parser.add_argument("--prompts", help="Comma-separated list of prompt IDs")
    plot_parser.add_argument("--model", help="Filter by model name")
    plot_parser.add_argument("--dataset", help="Filter by dataset name")
    plot_parser.add_argument("--output", help="Output file path")
    
    # Generate report command
    report_parser = subparsers.add_parser("report", help="Generate comprehensive report")
    report_parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = PromptTracker()
    
    # Process commands
    if args.command == "register":
        tags = args.tags.split(",") if args.tags else None
        prompt_id = tracker.register_prompt(
            args.prompt_file,
            name=args.name,
            description=args.description,
            parent_id=args.parent,
            tags=tags
        )
        print(f"Prompt registered with ID: {prompt_id}")
    
    elif args.command == "log":
        eval_id = tracker.log_evaluation(
            args.prompt_id,
            args.results_file,
            args.model,
            args.dataset,
            notes=args.notes
        )
        print(f"Evaluation logged with ID: {eval_id}")
    
    elif args.command == "list-prompts":
        prompts = tracker.list_prompts(
            tag=args.tag,
            min_accuracy=args.min_accuracy,
            limit=args.limit
        )
        
        if prompts:
            headers = ["ID", "Name", "Best Score", "Latest Score", "Evals", "Tags"]
            table_data = [
                [
                    p["prompt_id"],
                    p["name"],
                    f"{p['best_score'] * 100:.2f}%" if p['best_score'] else "N/A",
                    f"{p['latest_score'] * 100:.2f}%" if p['latest_score'] else "N/A",
                    p["eval_count"],
                    ", ".join(p["tags"])
                ]
                for p in prompts
            ]
            
            print(tabulate(table_data, headers=headers))
        else:
            print("No prompts found.")
    
    elif args.command == "list-evals":
        evals = tracker.list_evaluations(
            prompt_id=args.prompt,
            model=args.model,
            dataset=args.dataset,
            limit=args.limit
        )
        
        if evals:
            headers = ["Evaluation ID", "Prompt ID", "Model", "Dataset", "Accuracy", "Date"]
            table_data = []
            
            for e in evals:
                accuracy = e["metrics"].get("overall_accuracy", "N/A")
                if isinstance(accuracy, (int, float)):
                    accuracy = f"{accuracy * 100:.2f}%"
                
                table_data.append([
                    e["evaluation_id"],
                    e["prompt_id"],
                    e["model"],
                    e["dataset"],
                    accuracy,
                    e["timestamp"].split("T")[0]
                ])
            
            print(tabulate(table_data, headers=headers))
        else:
            print("No evaluations found.")
    
    elif args.command == "best":
        best = tracker.get_best_prompts(
            model=args.model,
            dataset=args.dataset,
            limit=args.limit
        )
        
        if best:
            headers = ["Prompt ID", "Name", "Model", "Dataset", "Accuracy", "Date"]
            table_data = [
                [
                    b["prompt_id"],
                    b["name"],
                    b["model"],
                    b["dataset"],
                    f"{b['accuracy'] * 100:.2f}%",
                    b["timestamp"].split("T")[0]
                ]
                for b in best
            ]
            
            print(tabulate(table_data, headers=headers))
        else:
            print("No best prompts found.")
    
    elif args.command == "plot":
        prompt_ids = args.prompts.split(",") if args.prompts else None
        tracker.plot_performance_trend(
            prompt_ids=prompt_ids,
            model=args.model,
            dataset=args.dataset,
            output_file=args.output
        )
    
    elif args.command == "report":
        report = tracker.generate_report(output_file=args.output)
        
        if not args.output:
            print(report)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()