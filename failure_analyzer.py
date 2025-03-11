#!/usr/bin/env python3
"""
Failure Analyzer for Sentence Extraction Evaluation

This module analyzes extraction failures to identify patterns and
categories that can be used to refine prompts.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from collections import Counter, defaultdict
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class FailureAnalyzer:
    """Analyzes extraction failures to identify patterns."""

    def __init__(
        self,
        results_path: str,
        output_dir: str = "failure_analysis",
        min_error_threshold: float = 0.5,
        cluster_count: int = 5
    ):
        """Initialize the failure analyzer.
        
        Args:
            results_path: Path to the evaluation results JSON file
            output_dir: Directory to store analysis results
            min_error_threshold: Minimum error score to classify as a failure (0.0-1.0)
            cluster_count: Number of clusters for sentence grouping
        """
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.min_error_threshold = min_error_threshold
        self.cluster_count = cluster_count
        self.results_data = None
        self.failures = []
        self.success_cases = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_results(self):
        """Load the evaluation results."""
        with open(self.results_path, 'r') as f:
            self.results_data = json.load(f)
            
    def identify_failures(self):
        """Identify and categorize extraction failures."""
        if not self.results_data:
            self.load_results()
            
        results = self.results_data.get("results", [])
        if not results:
            raise ValueError("No results found in the results file")
            
        self.failures = []
        self.success_cases = []
        
        for result in results:
            # Get score from item if available, otherwise compute it
            score = None
            sentence_info = None
            
            # Check if the metrics contain scores by sentence
            if "metrics" in self.results_data and "scores" in self.results_data["metrics"]:
                for s in self.results_data["metrics"]["scores"]["by_sentence"]:
                    if s["sentence"].startswith(result["sentence"][:50]):
                        sentence_info = s
                        score = s.get("score", 0.0)
                        break
            
            # If score not found, calculate manually
            if score is None:
                correct_values = 0
                total_values = 0
                
                expected = result.get("expected", {})
                extracted = result.get("extracted", {})
                
                if expected and extracted:
                    # Simple scoring - count matching keys
                    for key in expected:
                        total_values += 1
                        if key in extracted and extracted[key] == expected[key]:
                            correct_values += 1
                            
                score = correct_values / total_values if total_values > 0 else 0.0
            
            # Classify as failure or success
            if score < (1.0 - self.min_error_threshold):
                self.failures.append({
                    "sentence": result["sentence"],
                    "expected": result["expected"],
                    "extracted": result["extracted"],
                    "full_response": result.get("full_response", ""),
                    "score": score,
                    "sentence_info": sentence_info
                })
            else:
                self.success_cases.append({
                    "sentence": result["sentence"],
                    "expected": result["expected"],
                    "extracted": result["extracted"],
                    "full_response": result.get("full_response", ""),
                    "score": score,
                    "sentence_info": sentence_info
                })
                
        return self.failures
    
    def analyze_field_errors(self) -> Dict[str, Dict]:
        """Analyze which fields have the most extraction errors.
        
        Returns:
            Dictionary mapping field names to error statistics
        """
        if not self.failures:
            self.identify_failures()
            
        field_errors = defaultdict(lambda: {"missing": 0, "incorrect": 0, "total": 0})
        total_failures = len(self.failures)
        
        for failure in self.failures:
            expected = failure["expected"]
            extracted = failure["extracted"]
            
            for field, expected_value in expected.items():
                field_errors[field]["total"] += 1
                
                if field not in extracted:
                    field_errors[field]["missing"] += 1
                elif extracted[field] != expected_value:
                    field_errors[field]["incorrect"] += 1
        
        # Convert to regular dict and add percentages
        result = {}
        for field, stats in field_errors.items():
            result[field] = {
                "missing": stats["missing"],
                "incorrect": stats["incorrect"],
                "total": stats["total"],
                "missing_pct": stats["missing"] / stats["total"] * 100 if stats["total"] > 0 else 0,
                "incorrect_pct": stats["incorrect"] / stats["total"] * 100 if stats["total"] > 0 else 0,
                "overall_error_pct": (stats["missing"] + stats["incorrect"]) / stats["total"] * 100 if stats["total"] > 0 else 0,
                "occurrences_pct": stats["total"] / total_failures * 100
            }
            
        return result
    
    def cluster_failure_sentences(self) -> List[Dict]:
        """Cluster failure sentences to identify patterns.
        
        Returns:
            List of clusters with their sentences and characteristics
        """
        if not self.failures:
            self.identify_failures()
            
        if len(self.failures) < self.cluster_count:
            # Not enough failures to cluster meaningfully
            return [{"cluster_id": 0, "sentences": self.failures, "common_phrases": []}]
            
        # Extract sentences for clustering
        sentences = [f["sentence"] for f in self.failures]
        
        # Use TF-IDF vectorization for text features
        vectorizer = TfidfVectorizer(
            max_features=100, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Transform sentences to TF-IDF features
        features = vectorizer.fit_transform(sentences)
        
        # Adjust cluster count if needed
        actual_clusters = min(self.cluster_count, len(sentences) - 1)
        
        # Perform KMeans clustering
        kmeans = KMeans(
            n_clusters=actual_clusters,
            random_state=42,
            n_init=10
        )
        
        cluster_ids = kmeans.fit_predict(features)
        
        # Group failures by cluster
        clusters = defaultdict(list)
        for failure, cluster_id in zip(self.failures, cluster_ids):
            clusters[cluster_id].append(failure)
            
        # Find characteristic terms for each cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_centers = kmeans.cluster_centers_
        
        # Extract cluster characteristics
        cluster_results = []
        for cluster_id, failures in clusters.items():
            # Get top terms for this cluster
            center = cluster_centers[cluster_id]
            top_indices = center.argsort()[-10:][::-1]  # Top 10 terms
            top_terms = [feature_names[i] for i in top_indices]
            
            # Get common fields with errors
            field_errors = defaultdict(int)
            for failure in failures:
                expected = failure["expected"]
                extracted = failure["extracted"]
                
                for field in expected:
                    if field not in extracted or expected[field] != extracted[field]:
                        field_errors[field] += 1
            
            common_fields = sorted(
                field_errors.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]  # Top 5 fields
            
            cluster_results.append({
                "cluster_id": cluster_id,
                "size": len(failures),
                "sentences": failures,
                "common_terms": top_terms,
                "problematic_fields": common_fields,
                "avg_score": sum(f["score"] for f in failures) / len(failures)
            })
            
        # Sort clusters by size (largest first)
        return sorted(cluster_results, key=lambda x: x["size"], reverse=True)
    
    def extract_common_patterns(self) -> Dict[str, List[str]]:
        """Extract common linguistic patterns from failure cases.
        
        Returns:
            Dictionary of pattern categories and examples
        """
        if not self.failures:
            self.identify_failures()
            
        patterns = {
            "multiple_relations": [],
            "complex_temporal": [],
            "name_variations": [],
            "implicit_relations": [],
            "field_ambiguity": [],
            "enumeration": []
        }
        
        # Check for multiple relations pattern
        for failure in self.failures:
            sentence = failure["sentence"].lower()
            expected = failure["expected"]
            
            # Check for multiple relations (my X and Y)
            if ("my" in sentence and "and" in sentence and 
                any(rel in sentence for rel in ["father", "mother", "brother", "sister", "son", "daughter"])):
                
                # Check if exactly 2 people are mentioned
                if ("name_1" in expected and "name_2" in expected) or "multiple_people" in failure.get("full_response", "").lower():
                    patterns["multiple_relations"].append(failure["sentence"])
            
            # Check for complex temporal references
            if any(term in sentence for term in ["after", "before", "since", "until", "when", "while"]):
                if any(field in expected for field in ["year", "date", "time_period", "age", "duration"]):
                    patterns["complex_temporal"].append(failure["sentence"])
            
            # Check for name variations
            if ("dr." in sentence or "professor" in sentence or 
                any(title in sentence for title in ["colonel", "reverend", "lieutenant"])):
                if "name" in expected or "title" in expected:
                    patterns["name_variations"].append(failure["sentence"])
            
            # Check for implicit relations
            if "from my" in sentence or "on my" in sentence:
                if "relationship" in expected:
                    patterns["implicit_relations"].append(failure["sentence"])
            
            # Check for field ambiguity
            if "field_ambiguity" in failure.get("full_response", "").lower() or "unclear" in failure.get("full_response", "").lower():
                patterns["field_ambiguity"].append(failure["sentence"])
            
            # Check for enumeration pattern
            if any(num in sentence for num in ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]):
                if any(field.startswith("number_of") for field in expected):
                    patterns["enumeration"].append(failure["sentence"])
        
        # Limit to 5 examples per pattern and remove empty categories
        result = {}
        for pattern, examples in patterns.items():
            if examples:
                result[pattern] = examples[:5]  # Limit to 5 examples
                
        return result
    
    def compare_with_success_cases(self) -> Dict[str, Any]:
        """Compare failure cases with success cases to identify differences.
        
        Returns:
            Dictionary of key differences between failures and successes
        """
        if not self.failures or not self.success_cases:
            self.identify_failures()
            
        if not self.failures or not self.success_cases:
            return {"error": "Insufficient data for comparison"}
            
        # Analyze sentence length
        failure_lengths = [len(f["sentence"].split()) for f in self.failures]
        success_lengths = [len(s["sentence"].split()) for s in self.success_cases]
        
        avg_failure_length = sum(failure_lengths) / len(failure_lengths)
        avg_success_length = sum(success_lengths) / len(success_lengths)
        
        # Analyze complexity (number of commas as proxy)
        failure_commas = [f["sentence"].count(",") for f in self.failures]
        success_commas = [s["sentence"].count(",") for s in self.success_cases]
        
        avg_failure_commas = sum(failure_commas) / len(failure_commas)
        avg_success_commas = sum(success_commas) / len(success_commas)
        
        # Analyze number of fields
        failure_fields = [len(f["expected"]) for f in self.failures]
        success_fields = [len(s["expected"]) for s in self.success_cases]
        
        avg_failure_fields = sum(failure_fields) / len(failure_fields)
        avg_success_fields = sum(success_fields) / len(success_fields)
        
        # Finding unique fields in failures vs. successes
        failure_field_set = set(itertools.chain.from_iterable(
            f["expected"].keys() for f in self.failures
        ))
        
        success_field_set = set(itertools.chain.from_iterable(
            s["expected"].keys() for s in self.success_cases
        ))
        
        unique_to_failures = failure_field_set - success_field_set
        unique_to_successes = success_field_set - failure_field_set
        
        return {
            "sentence_complexity": {
                "avg_failure_length": avg_failure_length,
                "avg_success_length": avg_success_length,
                "length_difference_pct": (avg_failure_length / avg_success_length - 1) * 100 if avg_success_length > 0 else 0,
                "avg_failure_commas": avg_failure_commas,
                "avg_success_commas": avg_success_commas,
                "comma_difference_pct": (avg_failure_commas / avg_success_commas - 1) * 100 if avg_success_commas > 0 else 0
            },
            "field_complexity": {
                "avg_failure_fields": avg_failure_fields,
                "avg_success_fields": avg_success_fields,
                "field_difference_pct": (avg_failure_fields / avg_success_fields - 1) * 100 if avg_success_fields > 0 else 0
            },
            "unique_fields": {
                "unique_to_failures": list(unique_to_failures),
                "unique_to_successes": list(unique_to_successes)
            }
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive failure analysis report.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.failures:
            self.identify_failures()
            
        # Compile all analyses
        field_errors = self.analyze_field_errors()
        clusters = self.cluster_failure_sentences()
        patterns = self.extract_common_patterns()
        comparison = self.compare_with_success_cases()
        
        # Build report
        report = {
            "summary": {
                "total_results": len(self.failures) + len(self.success_cases),
                "failure_count": len(self.failures),
                "success_count": len(self.success_cases),
                "failure_rate": len(self.failures) / (len(self.failures) + len(self.success_cases)) * 100 if (len(self.failures) + len(self.success_cases)) > 0 else 0
            },
            "field_errors": field_errors,
            "sentence_clusters": clusters,
            "linguistic_patterns": patterns,
            "success_comparison": comparison,
            "prompt_improvement_areas": []
        }
        
        # Generate improvement recommendations
        improvement_areas = []
        
        # 1. Add recommendations for problematic fields
        problem_fields = sorted(
            field_errors.items(), 
            key=lambda x: x[1]["overall_error_pct"], 
            reverse=True
        )[:5]
        
        for field, stats in problem_fields:
            if stats["overall_error_pct"] > 50:
                area = {
                    "area": f"field_{field}",
                    "description": f"Improve extraction of '{field}' field",
                    "evidence": f"{stats['missing']} missing and {stats['incorrect']} incorrect out of {stats['total']} occurrences",
                    "suggestion": "Add explicit examples showing correct extraction of this field"
                }
                improvement_areas.append(area)
        
        # 2. Add recommendations for sentence patterns
        for pattern, examples in patterns.items():
            if len(examples) >= 2:  # At least 2 examples to consider it a pattern
                area = {
                    "area": f"pattern_{pattern}",
                    "description": f"Improve handling of {pattern.replace('_', ' ')} patterns",
                    "evidence": f"Found {len(examples)} examples of this pattern in failures",
                    "suggestion": "Add examples demonstrating correct extraction for this pattern",
                    "examples": examples[:2]  # Include 2 examples in the report
                }
                improvement_areas.append(area)
                
        # 3. Add recommendations from sentence complexity
        if comparison["sentence_complexity"]["length_difference_pct"] > 20:
            area = {
                "area": "complex_sentences",
                "description": "Improve handling of complex, longer sentences",
                "evidence": f"Failure cases are {comparison['sentence_complexity']['length_difference_pct']:.1f}% longer than successful cases",
                "suggestion": "Add step-by-step instructions for breaking down complex sentences"
            }
            improvement_areas.append(area)
            
        # 4. Add recommendations from field complexity
        if comparison["field_complexity"]["field_difference_pct"] > 20:
            area = {
                "area": "field_count",
                "description": "Improve handling of entries with many fields",
                "evidence": f"Failure cases have {comparison['field_complexity']['field_difference_pct']:.1f}% more fields than successful cases",
                "suggestion": "Add examples with many fields to demonstrate correct extraction"
            }
            improvement_areas.append(area)
            
        # 5. Add recommendations from unique fields
        for field in comparison["unique_fields"]["unique_to_failures"]:
            area = {
                "area": f"unique_field_{field}",
                "description": f"Improve extraction of rare field: '{field}'",
                "evidence": f"This field appears only in failure cases",
                "suggestion": "Add specific examples with this field to the prompt"
            }
            improvement_areas.append(area)
            
        report["prompt_improvement_areas"] = improvement_areas
        
        return report
    
    def save_analysis(self) -> str:
        """Save the analysis results to the output directory.
        
        Returns:
            Path to the generated report file
        """
        # Generate the report
        report = self.generate_report()
        
        # Create output file paths
        timestamp = Path(self.results_path).stem.split("_")[-1]
        output_base = f"failure_analysis_{timestamp}"
        
        report_path = self.output_dir / f"{output_base}_report.json"
        summary_path = self.output_dir / f"{output_base}_summary.md"
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Create human-readable summary
        with open(summary_path, 'w') as f:
            f.write("# Extraction Failure Analysis\n\n")
            
            # Summary statistics
            f.write("## Summary\n\n")
            f.write(f"- Total results analyzed: {report['summary']['total_results']}\n")
            f.write(f"- Failures: {report['summary']['failure_count']} ({report['summary']['failure_rate']:.1f}%)\n")
            f.write(f"- Successes: {report['summary']['success_count']}\n\n")
            
            # Field errors
            f.write("## Problematic Fields\n\n")
            problem_fields = sorted(
                report["field_errors"].items(), 
                key=lambda x: x[1]["overall_error_pct"], 
                reverse=True
            )[:5]
            
            for field, stats in problem_fields:
                f.write(f"### {field}\n\n")
                f.write(f"- Overall error rate: {stats['overall_error_pct']:.1f}%\n")
                f.write(f"- Missing: {stats['missing']} ({stats['missing_pct']:.1f}%)\n")
                f.write(f"- Incorrect: {stats['incorrect']} ({stats['incorrect_pct']:.1f}%)\n")
                f.write(f"- Total occurrences: {stats['total']}\n\n")
            
            # Linguistic patterns
            f.write("## Common Patterns in Failures\n\n")
            for pattern, examples in report["linguistic_patterns"].items():
                pattern_name = pattern.replace("_", " ").title()
                f.write(f"### {pattern_name}\n\n")
                f.write(f"Found {len(examples)} examples, including:\n\n")
                
                for example in examples[:3]:  # Show up to 3 examples
                    f.write(f"- \"{example}\"\n")
                f.write("\n")
            
            # Sentence complexity
            sc = report["success_comparison"]["sentence_complexity"]
            f.write("## Sentence Complexity Comparison\n\n")
            f.write(f"- Failed sentences avg length: {sc['avg_failure_length']:.1f} words (vs. {sc['avg_success_length']:.1f} for successes)\n")
            f.write(f"- Failed sentences avg commas: {sc['avg_failure_commas']:.1f} (vs. {sc['avg_success_commas']:.1f} for successes)\n\n")
            
            # Improvement recommendations
            f.write("## Prompt Improvement Recommendations\n\n")
            for i, area in enumerate(report["prompt_improvement_areas"]):
                f.write(f"### {i+1}. {area['description']}\n\n")
                f.write(f"**Evidence:** {area['evidence']}\n\n")
                f.write(f"**Suggestion:** {area['suggestion']}\n\n")
                
                if "examples" in area:
                    f.write("**Examples:**\n\n")
                    for example in area["examples"]:
                        f.write(f"- \"{example}\"\n")
                f.write("\n")
                
        print(f"Analysis saved to {report_path} and {summary_path}")
        return str(report_path)


def main():
    """Run the failure analyzer from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze extraction failures")
    parser.add_argument("results_file", help="Path to the evaluation results JSON file")
    parser.add_argument("--output-dir", default="failure_analysis", help="Directory to store analysis results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum error threshold to classify as failure")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for sentence grouping")
    
    args = parser.parse_args()
    
    analyzer = FailureAnalyzer(
        results_path=args.results_file,
        output_dir=args.output_dir,
        min_error_threshold=args.threshold,
        cluster_count=args.clusters
    )
    
    analyzer.save_analysis()


if __name__ == "__main__":
    main()