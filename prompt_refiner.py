#!/usr/bin/env python3
"""
Prompt Refiner for Sentence Extraction Evaluation

This module implements an iterative prompt refinement process
based on failure analysis, with validation to prevent overfitting.
"""

import os
import json
import time
import subprocess
import argparse
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
import datetime
import re


class PromptRefiner:
    """
    Implements iterative prompt refinement based on failure analysis.
    """
    
    def __init__(
        self,
        base_prompt_path: str,
        train_dataset_path: str,
        validation_dataset_path: str,
        eval_script_path: str = "claude_extraction_eval.py",
        failure_analyzer_path: str = "failure_analyzer.py",
        output_dir: str = "prompt_refinement",
        max_iterations: int = 5,
        model: str = "sonnet",
        improvement_threshold: float = 0.01,
        max_examples_to_add: int = 3
    ):
        """Initialize the prompt refiner.
        
        Args:
            base_prompt_path: Path to the initial prompt template
            train_dataset_path: Path to the training dataset
            validation_dataset_path: Path to the validation dataset
            eval_script_path: Path to the evaluation script
            failure_analyzer_path: Path to the failure analyzer script
            output_dir: Directory to store refined prompts and results
            max_iterations: Maximum number of refinement iterations
            model: Model to use for evaluation
            improvement_threshold: Minimum improvement required to continue refining
            max_examples_to_add: Maximum number of examples to add per iteration
        """
        self.base_prompt_path = Path(base_prompt_path)
        self.train_dataset_path = Path(train_dataset_path)
        self.validation_dataset_path = Path(validation_dataset_path)
        self.eval_script_path = Path(eval_script_path)
        self.failure_analyzer_path = Path(failure_analyzer_path)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.model = model
        self.improvement_threshold = improvement_threshold
        self.max_examples_to_add = max_examples_to_add
        
        self.iterations = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_evaluation(self, prompt_path: str, dataset_path: str) -> Tuple[Dict[str, Any], str]:
        """Run evaluation on a dataset using the given prompt.
        
        Args:
            prompt_path: Path to the prompt template
            dataset_path: Path to the dataset
            
        Returns:
            Tuple of (metrics dictionary, results file path)
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Prepare command
        cmd = [
            "python", str(self.eval_script_path),
            "--model", self.model,
            "--prompt-file", prompt_path,
            "--eval-data", dataset_path
        ]
        
        # Run evaluation
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse output to extract key metrics
            output_lines = result.stdout.split("\n")
            
            # Extract metrics from output
            metrics = {}
            for line in output_lines:
                if "Value accuracy" in line:
                    metrics["value_accuracy"] = float(line.split(":")[1].strip().rstrip("%")) / 100
                elif "Field extraction rate" in line:
                    metrics["field_extraction_rate"] = float(line.split(":")[1].strip().rstrip("%")) / 100
                elif "Average sentence score" in line:
                    metrics["avg_sentence_score"] = float(line.split(":")[1].strip().rstrip("%")) / 100
                elif "Successful extractions" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        success_part = parts[1].strip()
                        metrics["success_rate"] = float(success_part.split("(")[1].split("%")[0]) / 100
            
            # Find the output directory from stdout
            output_dir = None
            results_path = None
            
            for line in output_lines:
                if "Output directory:" in line:
                    output_dir = line.split("Output directory:")[1].strip()
                    break
            
            if output_dir:
                results_path = os.path.join(output_dir, "extraction_results.json")
            
            return metrics, results_path
            
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return {}, ""
    
    def analyze_failures(self, results_path: str) -> Dict[str, Any]:
        """Run failure analysis on evaluation results.
        
        Args:
            results_path: Path to the evaluation results JSON
            
        Returns:
            Dictionary with analysis results
        """
        if not results_path or not os.path.exists(results_path):
            print("No results file provided or file does not exist")
            return {}
            
        # Prepare command
        cmd = [
            "python", str(self.failure_analyzer_path),
            results_path,
            "--output-dir", str(self.output_dir / "analysis")
        ]
        
        # Run analyzer
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Find the analysis file path from output
            output_lines = result.stdout.split("\n")
            analysis_path = None
            
            for line in output_lines:
                if "Analysis saved to" in line:
                    parts = line.split("Analysis saved to")
                    if len(parts) > 1:
                        paths = parts[1].split("and")
                        if len(paths) > 0:
                            analysis_path = paths[0].strip()
                            break
            
            if analysis_path and os.path.exists(analysis_path):
                with open(analysis_path, 'r') as f:
                    analysis = json.load(f)
                    return analysis
            else:
                print("Could not find analysis file")
                return {}
                
        except subprocess.CalledProcessError as e:
            print(f"Error running failure analysis: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            return {}
    
    def refine_prompt(self, prompt_text: str, analysis: Dict[str, Any]) -> str:
        """Refine a prompt based on failure analysis.
        
        Args:
            prompt_text: Original prompt text
            analysis: Failure analysis results
            
        Returns:
            Refined prompt text
        """
        if not analysis or "prompt_improvement_areas" not in analysis:
            return prompt_text
            
        improvement_areas = analysis["prompt_improvement_areas"]
        if not improvement_areas:
            return prompt_text
            
        # Sort improvement areas by priority
        prioritized_areas = sorted(
            improvement_areas,
            key=lambda x: "field" in x["area"],  # Prioritize field improvements first
            reverse=True
        )[:self.max_examples_to_add]  # Limit to max examples per iteration
        
        # Prepare refinements
        refinements = []
        
        for area in prioritized_areas:
            # Create a refinement based on area type
            if "field_" in area["area"]:
                # Field-specific refinement
                field_name = area["area"].replace("field_", "")
                refinement = self._create_field_refinement(field_name, area, analysis)
                refinements.append(refinement)
                
            elif "pattern_" in area["area"]:
                # Pattern-specific refinement
                pattern_name = area["area"].replace("pattern_", "")
                refinement = self._create_pattern_refinement(pattern_name, area, analysis)
                refinements.append(refinement)
                
            elif "complex_sentences" in area["area"]:
                # Complexity refinement
                refinement = self._create_complexity_refinement(area, analysis)
                refinements.append(refinement)
                
            elif "unique_field_" in area["area"]:
                # Unique field refinement
                field_name = area["area"].replace("unique_field_", "")
                refinement = self._create_unique_field_refinement(field_name, area, analysis)
                refinements.append(refinement)
        
        # Apply refinements to the prompt
        refined_prompt = prompt_text
        
        # Find the appropriate insertion point 
        # Typically before the closing instructions or after the examples
        insertion_point = self._find_insertion_point(prompt_text)
        
        if insertion_point > 0:
            # Insert refinements at the identified point
            refined_prompt = (
                prompt_text[:insertion_point] +
                "\n\n# ADDITIONAL GUIDANCE FOR SPECIFIC CASES\n\n" +
                "\n\n".join(refinements) +
                "\n\n" +
                prompt_text[insertion_point:]
            )
        else:
            # Append to the end if no insertion point found
            refined_prompt = prompt_text + "\n\n# ADDITIONAL GUIDANCE FOR SPECIFIC CASES\n\n" + "\n\n".join(refinements)
        
        return refined_prompt
    
    def _create_field_refinement(self, field_name: str, area: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create a field-specific refinement.
        
        Args:
            field_name: The name of the field to refine
            area: The improvement area information
            analysis: The full analysis data
            
        Returns:
            Text block for the refined prompt
        """
        # Find examples for this field in failures
        field_examples = []
        for cluster in analysis.get("sentence_clusters", []):
            for sentence in cluster.get("sentences", [])[:2]:  # Limit to 2 examples per cluster
                expected = sentence.get("expected", {})
                if field_name in expected:
                    field_examples.append({
                        "sentence": sentence["sentence"],
                        "extracted_value": expected.get(field_name)
                    })
                    
        # Limit to 2 examples
        field_examples = field_examples[:2]
        
        # Create the refinement text
        refinement = f"## Pay special attention to the '{field_name}' field\n\n"
        refinement += f"{area['suggestion']}\n\n"
        
        if field_examples:
            refinement += "Examples:\n\n"
            for i, example in enumerate(field_examples):
                refinement += f"Example {i+1}: \"{example['sentence']}\"\n"
                refinement += f"The '{field_name}' field should be extracted as: \"{example['extracted_value']}\"\n\n"
                
        return refinement
    
    def _create_pattern_refinement(self, pattern_name: str, area: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create a pattern-specific refinement.
        
        Args:
            pattern_name: The name of the pattern to refine
            area: The improvement area information
            analysis: The full analysis data
            
        Returns:
            Text block for the refined prompt
        """
        pattern_display = pattern_name.replace("_", " ").title()
        
        # Get examples from the area
        examples = area.get("examples", [])
        
        # Create the refinement text
        refinement = f"## Handling {pattern_display} Patterns\n\n"
        refinement += f"{area['suggestion']}\n\n"
        
        if examples:
            refinement += "Examples:\n\n"
            for i, example in enumerate(examples):
                refinement += f"Example {i+1}: \"{example}\"\n\n"
                
                # Try to find expected output for this example
                found_expected = False
                for cluster in analysis.get("sentence_clusters", []):
                    for sentence in cluster.get("sentences", []):
                        if sentence["sentence"] == example:
                            refinement += "Expected extraction:\n"
                            refinement += "```json\n"
                            refinement += json.dumps(sentence["expected"], indent=2)
                            refinement += "\n```\n\n"
                            found_expected = True
                            break
                    if found_expected:
                        break
                
                # If no expected found, add a general note
                if not found_expected:
                    refinement += "For this type of sentence, pay special attention to correctly identifying relationships and chronological information.\n\n"
                
        return refinement
    
    def _create_complexity_refinement(self, area: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create a complexity-focused refinement.
        
        Args:
            area: The improvement area information
            analysis: The full analysis data
            
        Returns:
            Text block for the refined prompt
        """
        # Get examples of complex sentences
        complex_examples = []
        clusters = sorted(analysis.get("sentence_clusters", []), key=lambda x: len(x.get("sentences", [])[0]["sentence"]) if x.get("sentences") else 0, reverse=True)
        
        # Take the first sentence from the top 2 clusters with longest sentences
        for cluster in clusters[:2]:
            if cluster.get("sentences"):
                complex_examples.append(cluster["sentences"][0]["sentence"])
        
        # Create the refinement text
        refinement = "## Handling Complex, Longer Sentences\n\n"
        refinement += "When dealing with long, complex sentences that contain multiple pieces of information, follow these steps:\n\n"
        refinement += "1. First, identify all the family members mentioned in the sentence\n"
        refinement += "2. For each family member, extract their basic information (name, relationship)\n"
        refinement += "3. Then, systematically extract additional attributes from the sentence\n"
        refinement += "4. Pay attention to temporal markers (before, after, since, etc.) to establish chronology\n"
        refinement += "5. Ensure all relationships between family members are correctly captured\n\n"
        
        if complex_examples:
            refinement += "Examples of complex sentences:\n\n"
            for i, example in enumerate(complex_examples):
                refinement += f"Example {i+1}: \"{example}\"\n\n"
                
        return refinement
    
    def _create_unique_field_refinement(self, field_name: str, area: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Create a refinement for a unique/rare field.
        
        Args:
            field_name: The name of the field to refine
            area: The improvement area information
            analysis: The full analysis data
            
        Returns:
            Text block for the refined prompt
        """
        # Find examples for this field in failures
        field_examples = []
        for cluster in analysis.get("sentence_clusters", []):
            for sentence in cluster.get("sentences", []):
                expected = sentence.get("expected", {})
                if field_name in expected:
                    field_examples.append({
                        "sentence": sentence["sentence"],
                        "extracted_value": expected.get(field_name)
                    })
                    
        # Limit to 2 examples
        field_examples = field_examples[:2]
        
        # Create the refinement text
        refinement = f"## Special Field: '{field_name}'\n\n"
        refinement += f"Pay special attention to extracting the '{field_name}' field when applicable.\n\n"
        
        if field_examples:
            refinement += "Examples:\n\n"
            for i, example in enumerate(field_examples):
                refinement += f"Example {i+1}: \"{example['sentence']}\"\n"
                refinement += f"The '{field_name}' field should be extracted as: \"{example['extracted_value']}\"\n\n"
                
        return refinement
    
    def _find_insertion_point(self, prompt_text: str) -> int:
        """Find the appropriate insertion point in the prompt.
        
        Args:
            prompt_text: The prompt text
            
        Returns:
            Character index for insertion
        """
        # Try to find insertion points in order of preference
        
        # 1. Before "Remember to always include" or similar closing instructions
        closing_patterns = [
            "Remember to",
            "Finally,",
            "Always ensure",
            "In summary,",
            "To recap,",
            "IMPORTANT:"
        ]
        
        for pattern in closing_patterns:
            match = re.search(f"\n{pattern}", prompt_text)
            if match:
                return match.start()
        
        # 2. After examples section
        examples_end = prompt_text.find("\n\n", prompt_text.find("Example"))
        if examples_end > 0:
            return examples_end
        
        # 3. Before {SENTENCE} placeholder
        placeholder = prompt_text.find("{SENTENCE}")
        if placeholder > 0:
            # Look for a line break before the placeholder
            line_break = prompt_text.rfind("\n\n", 0, placeholder)
            if line_break > 0:
                return line_break
                
        # 4. Default to 2/3 of the way through the prompt
        return int(len(prompt_text) * 2/3)
    
    def run_refinement_process(self):
        """Run the iterative prompt refinement process."""
        # Load base prompt
        with open(self.base_prompt_path, 'r') as f:
            base_prompt = f.read()
            
        # Initialize tracking
        current_prompt = base_prompt
        current_prompt_path = self.base_prompt_path
        best_validation_score = 0
        iteration_results = []
        
        # Initial evaluation on validation set
        print(f"Running initial evaluation on validation set...")
        val_metrics, _ = self.run_evaluation(str(current_prompt_path), str(self.validation_dataset_path))
        
        if not val_metrics:
            print("Failed to run initial validation evaluation")
            return
            
        best_validation_score = val_metrics.get("value_accuracy", 0)
        
        print(f"Initial validation score: {best_validation_score:.4f}")
        
        # Track initial iteration
        iteration_results.append({
            "iteration": 0,
            "prompt_path": str(current_prompt_path),
            "train_metrics": None,  # No training metrics for initial prompt
            "validation_metrics": val_metrics,
            "analysis": None,  # No analysis for initial prompt
            "refinements": []
        })
        
        # Run iterations
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n=== Iteration {iteration} ===\n")
            
            # 1. Run evaluation on training set
            print(f"Running evaluation on training set...")
            train_metrics, train_results_path = self.run_evaluation(
                str(current_prompt_path), 
                str(self.train_dataset_path)
            )
            
            if not train_metrics:
                print("Failed to run training evaluation")
                break
                
            print(f"Training score: {train_metrics.get('value_accuracy', 0):.4f}")
            
            # 2. Analyze failures
            print(f"Analyzing failures...")
            analysis = self.analyze_failures(train_results_path)
            
            if not analysis:
                print("Failed to analyze failures")
                break
                
            # 3. Refine prompt
            print(f"Refining prompt...")
            refined_prompt = self.refine_prompt(current_prompt, analysis)
            
            # Check if any refinements were made
            if refined_prompt == current_prompt:
                print("No refinements needed/made")
                break
                
            # Save refined prompt
            refined_prompt_path = self.output_dir / f"refined_prompt_iteration_{iteration}.txt"
            with open(refined_prompt_path, 'w') as f:
                f.write(refined_prompt)
                
            # Extract refinements made
            refinements = []
            if "ADDITIONAL GUIDANCE FOR SPECIFIC CASES" in refined_prompt:
                additional_guidance = refined_prompt.split("# ADDITIONAL GUIDANCE FOR SPECIFIC CASES")[1]
                sections = re.findall(r'## ([^\n]+)', additional_guidance)
                refinements = sections[-self.max_examples_to_add:]  # Get only the new refinements
                
            print(f"Added {len(refinements)} refinements: {', '.join(refinements)}")
            
            # 4. Evaluate refined prompt on validation set
            print(f"Evaluating refined prompt on validation set...")
            val_metrics, _ = self.run_evaluation(
                str(refined_prompt_path), 
                str(self.validation_dataset_path)
            )
            
            if not val_metrics:
                print("Failed to run validation evaluation")
                break
                
            validation_score = val_metrics.get("value_accuracy", 0)
            print(f"Validation score: {validation_score:.4f} (previous best: {best_validation_score:.4f})")
            
            # 5. Check for improvement
            improvement = validation_score - best_validation_score
            
            iteration_results.append({
                "iteration": iteration,
                "prompt_path": str(refined_prompt_path),
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "analysis": {
                    "improvement_areas": analysis.get("prompt_improvement_areas", []),
                    "summary": analysis.get("summary", {})
                },
                "refinements": refinements,
                "improvement": improvement
            })
            
            if improvement >= self.improvement_threshold:
                print(f"Achieved improvement of {improvement:.4f}, continuing...")
                best_validation_score = validation_score
                current_prompt = refined_prompt
                current_prompt_path = refined_prompt_path
            else:
                print(f"Insufficient improvement ({improvement:.4f}). Stopping.")
                break
        
        # Store final results
        self.iterations = iteration_results
        self._save_results()
        
        # Return the best prompt path
        return current_prompt_path
    
    def _save_results(self):
        """Save the refinement process results."""
        if not self.iterations:
            print("No iterations to save")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        results_path = self.output_dir / f"refinement_results_{timestamp}.json"
        summary_path = self.output_dir / f"refinement_summary_{timestamp}.md"
        
        # Save JSON results
        with open(results_path, 'w') as f:
            json.dump({
                "base_prompt": str(self.base_prompt_path),
                "train_dataset": str(self.train_dataset_path),
                "validation_dataset": str(self.validation_dataset_path),
                "model": self.model,
                "iterations": self.iterations
            }, f, indent=2)
            
        # Create summary
        initial_val_score = 0
        final_val_score = 0
        best_prompt_path = str(self.base_prompt_path)
        all_refinements = []
        
        if self.iterations:
            initial_val_score = self.iterations[0].get("validation_metrics", {}).get("value_accuracy", 0)
            final_iteration = self.iterations[-1]
            final_val_score = final_iteration.get("validation_metrics", {}).get("value_accuracy", 0)
            best_prompt_path = final_iteration.get("prompt_path", str(self.base_prompt_path))
            
            # Collect all refinements
            for iteration in self.iterations:
                all_refinements.extend(iteration.get("refinements", []))
            
        # Write summary
        with open(summary_path, 'w') as f:
            f.write("# Prompt Refinement Summary\n\n")
            f.write(f"- **Model:** {self.model}\n")
            f.write(f"- **Base prompt:** {self.base_prompt_path}\n")
            f.write(f"- **Iterations completed:** {len(self.iterations) - 1}\n")
            f.write(f"- **Initial validation score:** {initial_val_score:.4f}\n")
            f.write(f"- **Final validation score:** {final_val_score:.4f}\n")
            f.write(f"- **Improvement:** {final_val_score - initial_val_score:.4f}\n")
            f.write(f"- **Best prompt:** {best_prompt_path}\n\n")
            
            f.write("## Refinements Added\n\n")
            if all_refinements:
                for i, refinement in enumerate(all_refinements):
                    f.write(f"{i+1}. {refinement}\n")
            else:
                f.write("No refinements added\n")
                
            f.write("\n## Iteration Details\n\n")
            
            for i, iteration in enumerate(self.iterations[1:], 1):
                f.write(f"### Iteration {i}\n\n")
                
                train_accuracy = iteration.get("train_metrics", {}).get("value_accuracy", 0)
                val_accuracy = iteration.get("validation_metrics", {}).get("value_accuracy", 0)
                improvement = iteration.get("improvement", 0)
                
                f.write(f"- **Train accuracy:** {train_accuracy:.4f}\n")
                f.write(f"- **Validation accuracy:** {val_accuracy:.4f}\n")
                f.write(f"- **Improvement:** {improvement:.4f}\n")
                
                refinements = iteration.get("refinements", [])
                if refinements:
                    f.write("- **Refinements added:**\n")
                    for refinement in refinements:
                        f.write(f"  - {refinement}\n")
                        
                f.write("\n")
            
        print(f"Results saved to {results_path} and {summary_path}")


def main():
    """Run the prompt refiner from command line."""
    parser = argparse.ArgumentParser(description="Refine prompts based on failure analysis")
    parser.add_argument("base_prompt", help="Path to the base prompt template")
    parser.add_argument("train_dataset", help="Path to the training dataset")
    parser.add_argument("validation_dataset", help="Path to the validation dataset")
    parser.add_argument("--eval-script", default="claude_extraction_eval.py", help="Path to the evaluation script")
    parser.add_argument("--analyzer-script", default="failure_analyzer.py", help="Path to the failure analyzer script")
    parser.add_argument("--output-dir", default="prompt_refinement", help="Directory to store refined prompts and results")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum number of refinement iterations")
    parser.add_argument("--model", default="sonnet", help="Model to use for evaluation")
    parser.add_argument("--improvement-threshold", type=float, default=0.01, help="Minimum improvement required to continue")
    parser.add_argument("--max-examples", type=int, default=3, help="Maximum number of examples to add per iteration")
    
    args = parser.parse_args()
    
    refiner = PromptRefiner(
        base_prompt_path=args.base_prompt,
        train_dataset_path=args.train_dataset,
        validation_dataset_path=args.validation_dataset,
        eval_script_path=args.eval_script,
        failure_analyzer_path=args.analyzer_script,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        model=args.model,
        improvement_threshold=args.improvement_threshold,
        max_examples_to_add=args.max_examples
    )
    
    best_prompt_path = refiner.run_refinement_process()
    print(f"Refinement complete. Best prompt: {best_prompt_path}")


if __name__ == "__main__":
    main()