#!/usr/bin/env python3
"""
Test chain-of-thought reasoning on a more complex sentence to see if it makes a difference
with more challenging examples.
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

def run_complex_test():
    """Run a test comparing standard vs chain-of-thought extraction on a complex example."""
    print("Running complex example comparison test...")
    
    # Use a complex example with nested relationships
    test_sentence = "My maternal great-grandmother, Elizabeth Parker Wilson, was a suffragette who marched in Washington DC in 1917 and was arrested three times for her activism before women gained the right to vote in 1920, and her daughter (my grandmother) Mary continued her legacy by becoming the first female professor at the state university in 1952."
    
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
    output_dir = Path("complex_cot_test_results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run standard prompt extraction
    print("\nRunning standard prompt extraction on complex sentence...")
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
    print("Running chain-of-thought prompt extraction on complex sentence...")
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
    print("\nComplex test complete!")
    print(f"Standard prompt processing time: {standard_time:.2f} seconds")
    print(f"Chain-of-thought processing time: {cot_time:.2f} seconds")
    print(f"Time difference: {cot_time - standard_time:.2f} seconds")
    
    # Compare family member counts
    standard_members = len(standard_json.get("family_members", []))
    cot_members = len(cot_json.get("family_members", []))
    print(f"\nStandard extraction found {standard_members} family members")
    print(f"Chain-of-thought extraction found {cot_members} family members")
    
    # Compare extraction details
    if standard_members > 0 and cot_members > 0:
        # Compare field counts for all members
        std_total_fields = sum(sum(1 for k, v in member.items() if v and k != "relation_to") 
                              for member in standard_json["family_members"])
        cot_total_fields = sum(sum(1 for k, v in member.items() if v and k != "relation_to") 
                              for member in cot_json["family_members"])
        
        print(f"\nTotal non-null fields (standard): {std_total_fields}")
        print(f"Total non-null fields (chain-of-thought): {cot_total_fields}")
        
        # Count relationship links
        std_relations = sum(len(member.get("relation_to", [])) for member in standard_json["family_members"])
        cot_relations = sum(len(member.get("relation_to", [])) for member in cot_json["family_members"])
        
        print(f"\nTotal relationship links (standard): {std_relations}")
        print(f"Total relationship links (chain-of-thought): {cot_relations}")
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    run_complex_test()