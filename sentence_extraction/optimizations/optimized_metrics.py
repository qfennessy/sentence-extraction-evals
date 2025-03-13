#!/usr/bin/env python
"""
Optimized metrics calculation for extraction evaluation.

This module provides optimized functions for calculating extraction metrics,
with early termination and selective computation based on configuration.
"""

import time
from typing import Dict, List, Any, Tuple, Set, Optional, Union

def normalize_structure_fast(data: Any) -> Any:
    """
    Normalize different data structures to allow for comparison, optimized for speed.
    
    Args:
        data: The data structure to normalize
        
    Returns:
        Normalized data structure
    """
    # Fast path: if data is None or empty, return as is
    if not data:
        return data
    
    # Handle flat structure (fast path)
    if isinstance(data, dict) and "family_members" not in data:
        # Check if this looks like test data
        test_data_keys = {"name", "relationship", "birthplace", "birthdate"}
        if test_data_keys.intersection(data.keys()):
            # Extract non-null fields
            return {k: v for k, v in data.items() if v is not None and k != "sentence"}
    
    # Handle family_members structure
    if isinstance(data, dict) and "family_members" in data:
        # Fast path: extract the first non-narrator family member
        all_fields = {}
        
        for member in data["family_members"]:
            if member.get("name") and member.get("name") != "narrator":
                # Found main person - use their fields
                all_fields["name"] = member["name"]
                
                # Copy other fields
                for k, v in member.items():
                    if k not in ["name", "relation_to"] and v is not None:
                        all_fields[k] = v
                
                # Extract relationship to narrator if available
                for relation in member.get("relation_to", []):
                    if relation.get("name") == "narrator":
                        all_fields["relationship"] = relation.get("relationship")
                        break
                
                # We found a main person, so stop here
                break
        
        return all_fields
    
    # Return unchanged if no normalization patterns match
    return data

def compare_extraction_results(expected: Any, extracted: Any, detailed_metrics: bool = False) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compare extracted results against expected values with optimizations.
    
    Args:
        expected: Expected values
        extracted: Extracted values
        detailed_metrics: Whether to compute detailed metrics (slower)
        
    Returns:
        Tuple of (score, errors_list)
    """
    # Fast path: if one is None/empty and the other isn't
    if bool(expected) != bool(extracted):
        return 0.0, [{"error": "Only one side has content"}]
    
    # Normalize the structures
    normalized_expected = normalize_structure_fast(expected)
    normalized_extracted = normalize_structure_fast(extracted)
    
    # Fast path: direct equality check for simple cases
    if normalized_expected == normalized_extracted:
        return 1.0, []
    
    # Fast path: if they're not both dictionaries, simple mismatch
    if not (isinstance(normalized_expected, dict) and isinstance(normalized_extracted, dict)):
        return 0.0, [{"error": "Type mismatch", "expected": type(normalized_expected), "extracted": type(normalized_extracted)}]
    
    # Detailed comparison
    errors = []
    matches = 0
    total_fields = len(normalized_expected)
    
    # Fast path: return early if no fields to check
    if total_fields == 0:
        return 1.0, []
    
    # Only examine expected fields (we don't penalize extra fields in extracted)
    for field, expected_value in normalized_expected.items():
        # When detailed_metrics is False, use faster core field checking
        if not detailed_metrics and field not in {"name", "relationship", "occupation", "birthplace", "birth_year"}:
            continue
            
        if field not in normalized_extracted:
            errors.append({
                "field": field,
                "expected": expected_value,
                "extracted": None,
                "status": "missing"
            })
            continue
            
        extracted_value = normalized_extracted[field]
        
        # String normalization for more forgiving comparison
        if isinstance(expected_value, str) and isinstance(extracted_value, str):
            expected_norm = expected_value.lower().strip()
            extracted_norm = extracted_value.lower().strip()
            
            # Exact match
            if expected_norm == extracted_norm:
                matches += 1
                continue
                
            # Partial match: one is subset of the other
            if expected_norm in extracted_norm or extracted_norm in expected_norm:
                # Partial credit for containing the value
                matches += 0.5
                errors.append({
                    "field": field,
                    "expected": expected_value,
                    "extracted": extracted_value,
                    "status": "partial_match"
                })
                continue
                
        # Direct value comparison for non-strings
        elif expected_value == extracted_value:
            matches += 1
            continue
            
        # Mismatch
        errors.append({
            "field": field,
            "expected": expected_value,
            "extracted": extracted_value,
            "status": "mismatch"
        })
    
    # Score is the ratio of matches to total fields
    score = matches / total_fields if total_fields > 0 else 0.0
    
    return score, errors

def calculate_metrics_optimized(results: List[Dict[str, Any]], detailed_metrics: bool = True) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from extraction results, optimized for speed.
    
    Args:
        results: List of extraction results to evaluate
        detailed_metrics: Whether to compute detailed metrics (slower)
        
    Returns:
        Dictionary with calculated metrics
    """
    start_time = time.time()
    
    metrics = {
        "total_sentences": len(results),
        "successful_extractions": 0,
        "failed_extractions": 0,
        "overall_value_accuracy": 0.0,
        "scores": {
            "by_sentence": [],
            "min_score": 0.0,
            "max_score": 0.0,
            "avg_score": 0.0,
            "median_score": 0.0
        }
    }
    
    # Fast path for empty results
    if not results:
        return metrics
    
    # Gather scores
    scores = []
    total_score = 0.0
    
    for result in results:
        # Check if extraction was successful
        if not result.get("extracted"):
            metrics["failed_extractions"] += 1
            scores.append(0.0)
            metrics["scores"]["by_sentence"].append({
                "sentence": result["sentence"][:50] + "..." if len(result["sentence"]) > 50 else result["sentence"],
                "score": 0.0
            })
            continue
            
        # Extraction was successful
        metrics["successful_extractions"] += 1
        
        # Compare expected and extracted
        score, errors = compare_extraction_results(
            result["expected"], 
            result["extracted"],
            detailed_metrics
        )
        
        scores.append(score)
        total_score += score
        
        # Add score to by_sentence list
        metrics["scores"]["by_sentence"].append({
            "sentence": result["sentence"][:50] + "..." if len(result["sentence"]) > 50 else result["sentence"],
            "score": score,
            "error_count": len(errors)
        })
    
    # Calculate score statistics
    if scores:
        metrics["overall_value_accuracy"] = total_score / len(scores)
        metrics["scores"]["min_score"] = min(scores)
        metrics["scores"]["max_score"] = max(scores)
        metrics["scores"]["avg_score"] = sum(scores) / len(scores)
        
        # Calculate median score
        sorted_scores = sorted(scores)
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 0:
            metrics["scores"]["median_score"] = (sorted_scores[mid-1] + sorted_scores[mid]) / 2
        else:
            metrics["scores"]["median_score"] = sorted_scores[mid]
    
    # Calculate overall field extraction rate
    metrics["overall_field_extraction_rate"] = metrics["successful_extractions"] / metrics["total_sentences"]
    
    # Add computation time
    metrics["computation_time"] = time.time() - start_time
    
    # Add detailed metrics only if requested
    if detailed_metrics:
        # Add field counts and extraction rates
        field_counts = {}
        extraction_rates = {}
        
        for result in results:
            if result.get("expected"):
                for field in result["expected"]:
                    field_counts[field] = field_counts.get(field, 0) + 1
                    
                    if result.get("extracted") and field in result["extracted"]:
                        extraction_rates[field] = extraction_rates.get(field, 0) + 1
        
        # Calculate extraction rates
        for field, count in extraction_rates.items():
            extraction_rates[field] = count / field_counts[field] if field_counts[field] > 0 else 0
        
        metrics["field_counts"] = field_counts
        metrics["extraction_rates_by_field"] = extraction_rates
    
    return metrics


def test_optimized_metrics():
    """Test the optimized metrics calculation functions."""
    # Test data
    test_results = [
        # Successful extraction
        {
            "sentence": "My father John is a doctor.",
            "expected": {"name": "John", "relationship": "father", "occupation": "doctor"},
            "extracted": {"name": "John", "relationship": "father", "occupation": "doctor"}
        },
        # Partial match
        {
            "sentence": "My sister Sarah works as a software engineer in Boston.",
            "expected": {"name": "Sarah", "relationship": "sister", "occupation": "software engineer", "location": "Boston"},
            "extracted": {"name": "Sarah", "relationship": "sister", "occupation": "software developer"}
        },
        # Failed extraction
        {
            "sentence": "My brother James graduated from MIT in 2015.",
            "expected": {"name": "James", "relationship": "brother", "education": "MIT", "graduation_year": "2015"},
            "extracted": {}
        }
    ]
    
    # Test normalization
    print("Testing structure normalization...")
    test_data = {
        "family_members": [
            {
                "name": "John",
                "occupation": "doctor",
                "relation_to": [
                    {"name": "narrator", "relationship": "father"}
                ]
            }
        ]
    }
    normalized = normalize_structure_fast(test_data)
    print(f"Normalized: {normalized}")
    
    # Test comparison
    print("\nTesting result comparison...")
    expected = {"name": "John", "relationship": "father", "occupation": "doctor"}
    extracted = {"name": "John", "relationship": "father", "occupation": "physician"}
    score, errors = compare_extraction_results(expected, extracted)
    print(f"Comparison score: {score}")
    print(f"Errors: {errors}")
    
    # Test metrics calculation
    print("\nTesting metrics calculation...")
    metrics_standard = calculate_metrics_optimized(test_results)
    print(f"Standard metrics computation time: {metrics_standard['computation_time']:.6f} seconds")
    
    metrics_fast = calculate_metrics_optimized(test_results, detailed_metrics=False)
    print(f"Fast metrics computation time: {metrics_fast['computation_time']:.6f} seconds")
    
    speedup = metrics_standard['computation_time'] / metrics_fast['computation_time'] if metrics_fast['computation_time'] > 0 else 0
    print(f"Speedup factor: {speedup:.2f}x")
    
    print("Optimized metrics tests completed.")


if __name__ == "__main__":
    test_optimized_metrics()