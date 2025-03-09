#!/usr/bin/env python3
"""
Simple test script to compare standard extraction with chain-of-thought extraction
on a single example sentence.
"""

import os
import json
import anthropic
from pathlib import Path
import time

def format_prompt(prompt_template_path, sentence):
    """Format the prompt template with the given sentence."""
    with open(prompt_template_path, 'r') as f:
        prompt_template = f.read()
    return prompt_template.replace("{SENTENCE}", sentence)

def extract_json_from_response(response):
    """Extract JSON from Claude's response."""
    try:
        # Look for JSON object in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        
        # If no JSON structure found, return empty dict
        return {}
    
    except json.JSONDecodeError:
        print(f"Failed to extract JSON from response")
        return {}

def run_simple_test():
    """Run a simple test comparing standard vs chain-of-thought extraction."""
    print("Running simple comparison test...")
    
    # Use a complex example for testing
    test_sentence = "My paternal grandparents, Harold and Margaret Jenkins, owned a dairy farm in Wisconsin for 47 years before selling it to my cousin Brian in 2005."
    
    # Load API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set")
        return
    
    # Initialize the client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Format both prompts
    standard_prompt = format_prompt("prompt_template.txt", test_sentence)
    cot_prompt = format_prompt("prompt_template_cot.txt", test_sentence)
    
    # Output directory for results
    output_dir = Path("simple_cot_test_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run standard prompt extraction
    print("\nRunning standard prompt extraction...")
    start_time = time.time()
    standard_response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0.0,
        system="You extract structured information from text about family members.",
        messages=[{"role": "user", "content": standard_prompt}]
    )
    standard_time = time.time() - start_time
    standard_text = standard_response.content[0].text
    standard_json = extract_json_from_response(standard_text)
    
    # Run CoT prompt extraction
    print("Running chain-of-thought prompt extraction...")
    start_time = time.time()
    cot_response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=4000,
        temperature=0.0,
        system="You extract structured information from text about family members.",
        messages=[{"role": "user", "content": cot_prompt}]
    )
    cot_time = time.time() - start_time
    cot_text = cot_response.content[0].text
    cot_json = extract_json_from_response(cot_text)
    
    # Save full responses
    with open(output_dir / "standard_response.txt", "w") as f:
        f.write(standard_text)
    
    with open(output_dir / "cot_response.txt", "w") as f:
        f.write(cot_text)
    
    # Save extracted JSON
    with open(output_dir / "standard_extraction.json", "w") as f:
        json.dump(standard_json, f, indent=2)
    
    with open(output_dir / "cot_extraction.json", "w") as f:
        json.dump(cot_json, f, indent=2)
    
    # Print summary
    print("\nTest complete!")
    print(f"Standard prompt processing time: {standard_time:.2f} seconds")
    print(f"Chain-of-thought processing time: {cot_time:.2f} seconds")
    print(f"Time difference: {cot_time - standard_time:.2f} seconds")
    
    # Compare family member counts
    standard_members = len(standard_json.get("family_members", []))
    cot_members = len(cot_json.get("family_members", []))
    print(f"\nStandard extraction found {standard_members} family members")
    print(f"Chain-of-thought extraction found {cot_members} family members")
    
    # Compare field counts in first person
    if standard_members > 0 and cot_members > 0:
        std_fields = sum(1 for k, v in standard_json["family_members"][0].items() if v and k != "relation_to")
        cot_fields = sum(1 for k, v in cot_json["family_members"][0].items() if v and k != "relation_to")
        print(f"\nNon-null fields in first person (standard): {std_fields}")
        print(f"Non-null fields in first person (chain-of-thought): {cot_fields}")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    run_simple_test()