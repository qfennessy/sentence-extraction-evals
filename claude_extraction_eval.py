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

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics for the extraction results."""
    metrics = {
        "total_sentences": len(results),
        "successful_extractions": 0,
        "failed_extractions": 0,
        "extraction_rates_by_field": {},
        "field_counts": {},
        "overall_field_extraction_rate": 0,
        "sentences_with_errors": []
    }
    
    total_fields = 0
    total_extracted_fields = 0
    
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
            continue
        
        metrics["successful_extractions"] += 1
        
        # Compare expected and extracted fields
        for field, expected_value in expected.items():
            # Count this field
            metrics["field_counts"][field] = metrics["field_counts"].get(field, 0) + 1
            total_fields += 1
            
            # Check if field was extracted
            if field in extracted:
                # Count successful extraction
                metrics["extraction_rates_by_field"][field] = metrics["extraction_rates_by_field"].get(field, 0) + 1
                total_extracted_fields += 1
    
    # Calculate extraction rates
    for field, count in metrics["extraction_rates_by_field"].items():
        metrics["extraction_rates_by_field"][field] = count / metrics["field_counts"][field]
    
    # Calculate overall field extraction rate
    if total_fields > 0:
        metrics["overall_field_extraction_rate"] = total_extracted_fields / total_fields
    
    return metrics

def save_results(results: List[Dict], metrics: Dict, output_file: str):
    """Save the evaluation results and metrics to a file."""
    output = {
        "results": results,
        "metrics": metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Also save a summary as CSV
    if metrics["extraction_rates_by_field"]:
        df = pd.DataFrame({
            "Field": list(metrics["extraction_rates_by_field"].keys()),
            "Extraction Rate": list(metrics["extraction_rates_by_field"].values()),
            "Count": [metrics["field_counts"][field] for field in metrics["extraction_rates_by_field"].keys()]
        })
        
        csv_file = output_file.replace('.json', '_summary.csv')
        df.sort_values("Extraction Rate", ascending=False).to_csv(csv_file, index=False)

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
    print(f"Overall field extraction rate: {metrics['overall_field_extraction_rate']*100:.2f}%")
    
    print("\nTop 5 best extracted fields:")
    fields_sorted = sorted(metrics["extraction_rates_by_field"].items(), key=lambda x: x[1], reverse=True)
    for field, rate in fields_sorted[:5]:
        print(f"  {field}: {rate*100:.2f}% ({metrics['field_counts'][field]} occurrences)")
    
    print("\nBottom 5 worst extracted fields:")
    for field, rate in fields_sorted[-5:]:
        print(f"  {field}: {rate*100:.2f}% ({metrics['field_counts'][field]} occurrences)")
    
    # Save results
    save_results(results, metrics, args.output_file)
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()