#!/usr/bin/env python
"""
This script evaluates Claude's ability to extract family details from sentences.
It takes a prompt template file and a JSON evaluation dataset as inputs.

Usage:
python claude_extraction_eval.py --prompt-file prompt_template.txt --eval-data eval_data.json --api-key your_api_key --output-file results.json
"""

import argparse
import json
import os
import time
import anthropic
import pandas as pd
from typing import Dict, List, Any, Optional
from tqdm import tqdm

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Claude's extraction capabilities")
    parser.add_argument("--prompt-file", required=True, help="Path to the prompt template file")
    parser.add_argument("--eval-data", required=True, help="Path to the evaluation data JSON file")
    parser.add_argument("--api-key", required=False, help="Anthropic API key (or set via ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-3-5-sonnet-20240620", help="Claude model to use")
    parser.add_argument("--output-file", default="extraction_results.json", help="Output file for results")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of sentences to evaluate in each batch")
    parser.add_argument("--max-sentences", type=int, default=None, help="Maximum number of sentences to evaluate")
    
    return parser.parse_args()

def load_data(prompt_file: str, eval_data_file: str, max_sentences: Optional[int] = None) -> tuple:
    """Load the prompt template and evaluation data."""
    # Load prompt template
    with open(prompt_file, 'r') as f:
        prompt_template = f.read().strip()
    
    # Load evaluation data
    with open(eval_data_file, 'r') as f:
        eval_data = json.load(f)
    
    # Extract sentences and their extracted information
    sentences = eval_data.get("sentences", [])
    
    # Limit number of sentences if specified
    if max_sentences is not None:
        sentences = sentences[:max_sentences]
    
    return prompt_template, sentences

def format_prompt(prompt_template: str, sentence: str) -> str:
    """Format the prompt template with the given sentence."""
    return prompt_template.replace("{SENTENCE}", sentence)

def extract_json_from_response(response: str) -> Dict:
    """Extract JSON from Claude's response."""
    try:
        # Look for JSON object in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        else:
            # Try to find JSON-like content with different formatting
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            json_lines = []
            recording = False
            
            for line in lines:
                if line.startswith('{'):
                    recording = True
                    
                if recording:
                    json_lines.append(line)
                    
                if line.endswith('}'):
                    recording = False
            
            if json_lines:
                json_str = ' '.join(json_lines)
                return json.loads(json_str)
        
        # If no JSON structure found, return empty dict
        return {}
    
    except json.JSONDecodeError:
        print(f"Failed to extract JSON from response: {response[:100]}...")
        return {}

def evaluate_extraction(client, prompt_template: str, sentences: List[Dict], 
                       model: str, batch_size: int = 5) -> List[Dict]:
    """Evaluate Claude's extraction capabilities on the given sentences."""
    results = []
    
    # Process sentences in batches
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        for item in tqdm(batch, desc=f"Processing batch {i//batch_size + 1}"):
            sentence = item["sentence"]
            expected = item["extracted_information"]
            
            # Format prompt with the sentence
            prompt = format_prompt(prompt_template, sentence)
            
            try:
                # Call Claude API
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    temperature=0,
                    system="You extract structured information from text about family members.",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract response content
                response_text = response.content[0].text
                
                # Parse the extracted information from the response
                extracted = extract_json_from_response(response_text)
                
                # Store results
                results.append({
                    "sentence": sentence,
                    "expected": expected,
                    "extracted": extracted,
                    "full_response": response_text
                })
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing sentence: {sentence[:50]}...")
                print(f"Error: {str(e)}")
                results.append({
                    "sentence": sentence,
                    "expected": expected,
                    "extracted": {},
                    "error": str(e),
                    "full_response": ""
                })
                time.sleep(1)  # Longer delay after an error
    
    return results

def normalize_structure(data):
    """
    Normalize different data structures to allow for comparison.
    Handles format differences between test data and the model's output.
    """
    if not data:
        return data
    
    # If it's the expected flat format from test data
    if isinstance(data, dict) and "family_members" not in data:
        # Check for keys that indicate this is a flat structure from the test data
        test_data_keys = ["name", "relationship", "birthplace", "birthdate", 
                          "military_service", "occupation", "deathplace"]
        is_test_data = any(key in data for key in test_data_keys)
        
        if is_test_data:
            # Extract all fields from the flat structure
            extracted_fields = {}
            for k, v in data.items():
                # Skip null values
                if v is not None:
                    extracted_fields[k] = v
            return extracted_fields
    
    # If it's already in the family_members format
    if isinstance(data, dict) and "family_members" in data:
        # Extract all unique fields from all family members
        all_fields = {}
        for member in data["family_members"]:
            # Get name and other basic attributes
            if "name" in member and member["name"]:
                # Skip 'narrator' as it's added automatically
                if member["name"] == "narrator":
                    continue
                    
                all_fields["name"] = member["name"]
                
                # Get other fields
                for k, v in member.items():
                    if k not in ["name", "relation_to"] and v is not None:
                        all_fields[k] = v
                
                # Look at relationships
                if "relation_to" in member and member["relation_to"]:
                    for relation in member["relation_to"]:
                        if "name" in relation and relation["name"] != "narrator":
                            if "relationship" in relation:
                                rel_type = relation["relationship"].lower()
                                # Map relationship fields
                                if rel_type in ["father", "mother", "parent"]:
                                    all_fields["relationship"] = rel_type.capitalize()
                                elif rel_type in ["brother", "sister", "sibling"]:
                                    all_fields["relationship"] = "Sibling"
                                elif rel_type in ["son", "daughter", "child"]:
                                    all_fields["relationship"] = "Child"
                                elif rel_type in ["uncle", "aunt"]:
                                    all_fields["relationship"] = "Uncle/Aunt"
                                elif rel_type in ["grandfather", "grandmother", "grandparent"]:
                                    all_fields["relationship"] = "Grandparent"
                                # Add other relationship mappings as needed
        
        return all_fields
    
    # Return unchanged if no normalization needed
    return data

def deep_compare(expected, extracted, field_path=""):
    """
    Recursively compare expected and extracted values, handling nested structures.
    Returns a tuple of (num_correct, num_total, details) where details is a list of 
    mismatches with their paths.
    """
    # Normalize the data structures for comparison
    normalized_expected = normalize_structure(expected)
    normalized_extracted = normalize_structure(extracted)
    
    correct = 0
    total = 0
    details = []
    
    # If both are dictionaries, compare their keys and values
    if isinstance(normalized_expected, dict) and isinstance(normalized_extracted, dict):
        # Count each key-value pair in expected
        for key, exp_value in normalized_expected.items():
            total += 1
            current_path = f"{field_path}.{key}" if field_path else key
            
            if key in normalized_extracted:
                ext_value = normalized_extracted[key]
                # Recursively compare nested values
                sub_correct, sub_total, sub_details = deep_compare(exp_value, ext_value, current_path)
                correct += sub_correct
                total += sub_total - 1  # Subtract 1 because we already counted the key existence
                details.extend(sub_details)
            else:
                details.append({
                    "path": current_path,
                    "expected": exp_value,
                    "extracted": None,
                    "status": "missing"
                })
    
    # If both are lists, compare their elements
    elif isinstance(normalized_expected, list) and isinstance(normalized_extracted, list):
        # For lists, we'll try to match items in a way that maximizes matches
        # This is a simplified approach - could be improved with more complex matching
        remaining_extracted = normalized_extracted.copy()
        
        for i, exp_item in enumerate(normalized_expected):
            total += 1
            current_path = f"{field_path}[{i}]"
            
            best_match = None
            best_match_score = -1
            best_match_index = -1
            
            # Find the best matching item in the extracted list
            for j, ext_item in enumerate(remaining_extracted):
                sub_correct, sub_total, _ = deep_compare(exp_item, ext_item)
                match_score = sub_correct / max(1, sub_total) if sub_total > 0 else 0
                
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match = ext_item
                    best_match_index = j
            
            # If we found a reasonable match
            if best_match_score > 0.5:
                sub_correct, sub_total, sub_details = deep_compare(exp_item, best_match, current_path)
                correct += 1  # Count the item match
                # Add nested comparison details
                details.extend(sub_details)
                # Remove the matched item so we don't match it again
                if best_match_index >= 0:
                    del remaining_extracted[best_match_index]
            else:
                details.append({
                    "path": current_path,
                    "expected": exp_item,
                    "extracted": "No good match found",
                    "status": "mismatch"
                })
    
    # For primitive values, direct comparison
    else:
        total = 1
        # Normalize values for comparison (cast to string, lowercase)
        exp_norm = str(normalized_expected).lower() if normalized_expected is not None else ""
        ext_norm = str(normalized_extracted).lower() if normalized_extracted is not None else ""
        
        if exp_norm == ext_norm:
            correct = 1
        # Partial credit for numeric values with small differences
        elif (isinstance(normalized_expected, (int, float)) and isinstance(normalized_extracted, (int, float)) and 
              abs(normalized_expected - normalized_extracted) / max(1, abs(normalized_expected)) < 0.1):
            correct = 0.8
            details.append({
                "path": field_path,
                "expected": normalized_expected,
                "extracted": normalized_extracted,
                "status": "partial_match"
            })
        # Partial credit for text with significant overlap
        elif (isinstance(normalized_expected, str) and isinstance(normalized_extracted, str)):
            # Check if one is contained in the other
            if ext_norm in exp_norm or exp_norm in ext_norm:
                correct = 0.8
                details.append({
                    "path": field_path,
                    "expected": normalized_expected,
                    "extracted": normalized_extracted,
                    "status": "partial_match"
                })
            # Check for major overlap
            elif len(exp_norm) > 0 and len(ext_norm) > 0:
                # Calculate word overlap
                exp_words = set(exp_norm.split())
                ext_words = set(ext_norm.split())
                common_words = exp_words.intersection(ext_words)
                
                if len(common_words) > 0:
                    # Calculate Jaccard similarity
                    similarity = len(common_words) / len(exp_words.union(ext_words))
                    if similarity > 0.5:
                        correct = similarity
                        details.append({
                            "path": field_path,
                            "expected": normalized_expected,
                            "extracted": normalized_extracted,
                            "status": "partial_match",
                            "similarity": similarity
                        })
                    else:
                        correct = 0
                        details.append({
                            "path": field_path,
                            "expected": normalized_expected, 
                            "extracted": normalized_extracted,
                            "status": "mismatch"
                        })
                else:
                    correct = 0
                    details.append({
                        "path": field_path,
                        "expected": normalized_expected, 
                        "extracted": normalized_extracted,
                        "status": "mismatch"
                    })
            else:
                correct = 0
                details.append({
                    "path": field_path,
                    "expected": normalized_expected, 
                    "extracted": normalized_extracted,
                    "status": "mismatch"
                })
        else:
            correct = 0
            details.append({
                "path": field_path,
                "expected": normalized_expected, 
                "extracted": normalized_extracted,
                "status": "mismatch"
            })
    
    return correct, total, details

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics for the extraction results."""
    metrics = {
        "total_sentences": len(results),
        "successful_extractions": 0,
        "failed_extractions": 0,
        "extraction_rates_by_field": {},
        "field_counts": {},
        "overall_field_extraction_rate": 0,
        "sentences_with_errors": [],
        "scores": {
            "by_sentence": [],
            "min_score": 0.0,
            "max_score": 0.0,
            "avg_score": 0.0,
            "median_score": 0.0,
            "total_score": 0.0
        },
        "field_accuracy": {},
        "error_details": []
    }
    
    total_fields = 0
    total_extracted_fields = 0
    total_correct_values = 0
    total_expected_values = 0
    scores = []
    field_errors = {}
    
    for result in results:
        expected = result["expected"]
        extracted = result["extracted"]
        
        # Check if any fields were extracted
        if not extracted:
            metrics["failed_extractions"] += 1
            metrics["sentences_with_errors"].append({
                "sentence": result["sentence"],
                "error": "No fields extracted"
            })
            
            # Score is 0 for failed extraction
            sentence_score = 0.0
            scores.append(sentence_score)
            metrics["scores"]["by_sentence"].append({
                "sentence": result["sentence"][:50] + "..." if len(result["sentence"]) > 50 else result["sentence"],
                "score": sentence_score,
                "correct_values": 0,
                "total_values": len(expected) if isinstance(expected, dict) else 0,
                "error_details": ["No extraction"]
            })
            continue
        
        metrics["successful_extractions"] += 1
        
        # Deep comparison of expected and extracted structures
        sentence_correct, sentence_total, error_details = deep_compare(expected, extracted)
        
        # Accumulate totals for overall accuracy
        total_correct_values += sentence_correct
        total_expected_values += sentence_total
        
        # Track errors by field
        for error in error_details:
            field = error["path"].split(".")[0] if "." in error["path"] else error["path"].split("[")[0]
            if field not in field_errors:
                field_errors[field] = []
            field_errors[field].append(error)
        
        # Calculate score for this sentence
        sentence_score = sentence_correct / sentence_total if sentence_total > 0 else 0.0
        scores.append(sentence_score)
        
        # Count basic field presence for backward compatibility
        sentence_extracted_fields = 0
        sentence_total_fields = 0
        for field in expected:
            # Count this field
            metrics["field_counts"][field] = metrics["field_counts"].get(field, 0) + 1
            total_fields += 1
            sentence_total_fields += 1
            
            # Check if field was extracted
            if field in extracted:
                # Count successful extraction
                metrics["extraction_rates_by_field"][field] = metrics["extraction_rates_by_field"].get(field, 0) + 1
                total_extracted_fields += 1
                sentence_extracted_fields += 1
        
        # Store detailed sentence score info
        metrics["scores"]["by_sentence"].append({
            "sentence": result["sentence"][:50] + "..." if len(result["sentence"]) > 50 else result["sentence"],
            "score": sentence_score,
            "correct_values": sentence_correct,
            "total_values": sentence_total,
            "extracted_fields": sentence_extracted_fields,
            "total_fields": sentence_total_fields,
            "error_count": len(error_details)
        })
        
        # Store detailed error information
        if error_details:
            metrics["error_details"].append({
                "sentence": result["sentence"][:100] + "..." if len(result["sentence"]) > 100 else result["sentence"],
                "errors": error_details[:10]  # Limit to first 10 errors to keep the output manageable
            })
    
    # Calculate extraction rates (field presence)
    for field, count in metrics["extraction_rates_by_field"].items():
        metrics["extraction_rates_by_field"][field] = count / metrics["field_counts"][field]
    
    # Calculate overall field extraction rate (presence only)
    if total_fields > 0:
        metrics["overall_field_extraction_rate"] = total_extracted_fields / total_fields
    
    # Calculate overall value accuracy (content correctness)
    metrics["overall_value_accuracy"] = total_correct_values / total_expected_values if total_expected_values > 0 else 0.0
    
    # Calculate field-specific accuracy
    for field, errors in field_errors.items():
        total_instances = metrics["field_counts"].get(field, 0)
        error_count = len(errors)
        if total_instances > 0:
            metrics["field_accuracy"][field] = 1.0 - (error_count / total_instances)
        else:
            metrics["field_accuracy"][field] = 0.0
    
    # Calculate score statistics
    if scores:
        metrics["scores"]["min_score"] = min(scores)
        metrics["scores"]["max_score"] = max(scores)
        metrics["scores"]["avg_score"] = sum(scores) / len(scores)
        metrics["scores"]["total_score"] = metrics["overall_value_accuracy"]  # Use value accuracy as total score
        
        # Calculate median score
        sorted_scores = sorted(scores)
        mid = len(sorted_scores) // 2
        if len(sorted_scores) % 2 == 0:
            metrics["scores"]["median_score"] = (sorted_scores[mid-1] + sorted_scores[mid]) / 2
        else:
            metrics["scores"]["median_score"] = sorted_scores[mid]
    
    return metrics

def save_results(results: List[Dict], metrics: Dict, output_file: str):
    """Save the evaluation results and metrics to a file."""
    output = {
        "results": results,
        "metrics": metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save scores by sentence to CSV with detailed metrics
    if metrics["scores"]["by_sentence"]:
        scores_df = pd.DataFrame(metrics["scores"]["by_sentence"])
        scores_csv_file = output_file.replace('.json', '_scores.csv')
        scores_df.sort_values("score", ascending=False).to_csv(scores_csv_file, index=False)
    
    # Save field accuracy to CSV (more detailed than just presence/absence)
    if metrics.get("field_accuracy"):
        field_accuracy_data = []
        for field, accuracy in metrics["field_accuracy"].items():
            presence_rate = metrics["extraction_rates_by_field"].get(field, 0)
            count = metrics["field_counts"].get(field, 0)
            field_accuracy_data.append({
                "Field": field,
                "Content Accuracy": accuracy,
                "Presence Rate": presence_rate,
                "Count": count
            })
        
        field_accuracy_df = pd.DataFrame(field_accuracy_data)
        accuracy_csv_file = output_file.replace('.json', '_field_accuracy.csv')
        field_accuracy_df.sort_values("Content Accuracy", ascending=False).to_csv(accuracy_csv_file, index=False)
    
    # Save extraction rates by field to CSV (simple presence/absence)
    if metrics["extraction_rates_by_field"]:
        fields_df = pd.DataFrame({
            "Field": list(metrics["extraction_rates_by_field"].keys()),
            "Extraction Rate": list(metrics["extraction_rates_by_field"].values()),
            "Count": [metrics["field_counts"][field] for field in metrics["extraction_rates_by_field"].keys()]
        })
        
        # Create main summary CSV with both field stats and overall scores
        csv_file = output_file.replace('.json', '_summary.csv')
        fields_df.sort_values("Extraction Rate", ascending=False).to_csv(csv_file, index=False)
        
        # Append detailed score summary to the CSV
        with open(csv_file, 'a') as f:
            f.write("\n\nOverall Score Statistics\n")
            f.write(f"Value Accuracy (Total Score),{metrics.get('overall_value_accuracy', 0):.4f}\n")
            f.write(f"Field Extraction Rate,{metrics['overall_field_extraction_rate']:.4f}\n")
            f.write(f"Average Sentence Score,{metrics['scores']['avg_score']:.4f}\n")
            f.write(f"Median Sentence Score,{metrics['scores']['median_score']:.4f}\n")
            f.write(f"Min Sentence Score,{metrics['scores']['min_score']:.4f}\n")
            f.write(f"Max Sentence Score,{metrics['scores']['max_score']:.4f}\n")
            f.write(f"Successful Extractions,{metrics['successful_extractions']}\n")
            f.write(f"Failed Extractions,{metrics['failed_extractions']}\n")
            
    # Save detailed error examples to a text file for analysis
    if metrics.get("error_details"):
        error_file = output_file.replace('.json', '_errors.txt')
        with open(error_file, 'w') as f:
            f.write(f"Detailed Error Analysis\n{'='*50}\n\n")
            for i, error_item in enumerate(metrics["error_details"]):
                f.write(f"Sentence {i+1}: {error_item['sentence']}\n\n")
                for j, error in enumerate(error_item['errors']):
                    f.write(f"  Error {j+1}: {error['path']} - {error['status']}\n")
                    f.write(f"    Expected: {error['expected']}\n")
                    f.write(f"    Extracted: {error['extracted']}\n\n")
                f.write(f"{'-'*50}\n\n")

def main():
    args = parse_arguments()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api-key or ANTHROPIC_API_KEY environment variable")
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Load data
    prompt_template, sentences = load_data(args.prompt_file, args.eval_data, args.max_sentences)
    
    print(f"Loaded {len(sentences)} sentences for evaluation")
    print(f"Using model: {args.model}")
    
    # Evaluate extraction
    results = evaluate_extraction(
        client, 
        prompt_template, 
        sentences, 
        args.model, 
        args.batch_size
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total sentences: {metrics['total_sentences']}")
    print(f"Successful extractions: {metrics['successful_extractions']} ({metrics['successful_extractions']/metrics['total_sentences']*100:.2f}%)")
    print(f"Failed extractions: {metrics['failed_extractions']} ({metrics['failed_extractions']/metrics['total_sentences']*100:.2f}%)")
    
    # Print detailed score statistics
    print("\nDetailed Score Statistics:")
    print(f"  Value accuracy (total score): {metrics.get('overall_value_accuracy', 0)*100:.2f}%")
    print(f"  Field extraction rate: {metrics['overall_field_extraction_rate']*100:.2f}%")
    print(f"  Average sentence score: {metrics['scores']['avg_score']*100:.2f}%")
    print(f"  Median sentence score: {metrics['scores']['median_score']*100:.2f}%")
    print(f"  Min sentence score: {metrics['scores']['min_score']*100:.2f}%")
    print(f"  Max sentence score: {metrics['scores']['max_score']*100:.2f}%")
    
    # Print field accuracy (content correctness)
    if metrics.get("field_accuracy"):
        print("\nTop 5 most accurate fields (content correctness):")
        accuracy_sorted = sorted(metrics["field_accuracy"].items(), key=lambda x: x[1], reverse=True)
        for field, accuracy in accuracy_sorted[:5]:
            count = metrics["field_counts"].get(field, 0)
            print(f"  {field}: {accuracy*100:.2f}% correct ({count} occurrences)")
        
        if len(accuracy_sorted) > 5:
            print("\nBottom 5 least accurate fields (content correctness):")
            for field, accuracy in accuracy_sorted[-5:]:
                count = metrics["field_counts"].get(field, 0)
                print(f"  {field}: {accuracy*100:.2f}% correct ({count} occurrences)")
    
    # Print top 3 and bottom 3 scoring sentences
    scores_sorted = sorted(metrics["scores"]["by_sentence"], key=lambda x: x["score"], reverse=True)
    print("\nTop 3 best performing sentences:")
    for item in scores_sorted[:3]:
        correct = item.get("correct_values", 0)
        total = item.get("total_values", 1)
        print(f"  Score: {item['score']*100:.2f}% ({correct}/{total} correct values) - \"{item['sentence']}\"")
    
    print("\nBottom 3 worst performing sentences:")
    for item in scores_sorted[-3:]:
        correct = item.get("correct_values", 0)
        total = item.get("total_values", 1)
        print(f"  Score: {item['score']*100:.2f}% ({correct}/{total} correct values) - \"{item['sentence']}\"")
    
    # Print field presence rates (for backward compatibility)
    print("\nTop 5 most frequently extracted fields (presence only):")
    fields_sorted = sorted(metrics["extraction_rates_by_field"].items(), key=lambda x: x[1], reverse=True)
    for field, rate in fields_sorted[:5]:
        print(f"  {field}: {rate*100:.2f}% present ({metrics['field_counts'][field]} occurrences)")
    
    if len(fields_sorted) > 5:
        print("\nBottom 5 least frequently extracted fields (presence only):")
        for field, rate in fields_sorted[-5:]:
            print(f"  {field}: {rate*100:.2f}% present ({metrics['field_counts'][field]} occurrences)")
    
    # Print error summary
    if metrics.get("error_details"):
        error_count = sum(len(item["errors"]) for item in metrics["error_details"])
        error_types = {}
        for item in metrics["error_details"]:
            for error in item["errors"]:
                status = error["status"]
                error_types[status] = error_types.get(status, 0) + 1
        
        print("\nError Summary:")
        print(f"  Total errors found: {error_count}")
        print("  Error types:")
        for status, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"    {status}: {count} ({count/error_count*100:.1f}%)")
    
    # Save results
    save_results(results, metrics, args.output_file)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()