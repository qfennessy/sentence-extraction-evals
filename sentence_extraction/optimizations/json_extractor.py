#!/usr/bin/env python
"""
Optimized JSON extraction for handling model responses.

This module provides optimized functions for extracting JSON objects from
text responses, with better pattern matching and fallback mechanisms.
"""

import re
import json
from typing import Dict, Any, Optional

# Compile regular expression patterns once for better performance
JSON_PATTERN = re.compile(r'({[\s\S]*})')
JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*({[\s\S]*?})\s*```')
JSON_LINES_PATTERN = re.compile(r'(?:^|\n)(\{.*\})(?:$|\n)')

def extract_json_efficiently(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text using optimized pattern matching.
    
    This function uses a series of increasingly sophisticated extraction
    methods, starting with the fastest and falling back to more complex
    methods if needed.
    
    Args:
        text: The text response to extract JSON from
        
    Returns:
        Extracted JSON as a dictionary, or empty dict if extraction fails
    """
    if not text:
        return {}
    
    # Method 1: Try markdown code block extraction (fastest)
    json_obj = _extract_from_code_block(text)
    if json_obj:
        return json_obj
    
    # Method 2: Try direct JSON pattern match (fast)
    json_obj = _extract_using_json_pattern(text)
    if json_obj:
        return json_obj
    
    # Method 3: Try line-by-line scanning (medium)
    json_obj = _extract_from_json_lines(text)
    if json_obj:
        return json_obj
    
    # Method 4: Try comprehensive brute force method (slow)
    json_obj = _extract_json_comprehensive(text)
    if json_obj:
        return json_obj
    
    # Return empty dict if all methods fail
    return {}

def _extract_from_code_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from a markdown code block.
    
    Args:
        text: Text to extract from
        
    Returns:
        Extracted JSON dict or None if extraction fails
    """
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None

def _extract_using_json_pattern(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON using a simple pattern match.
    
    Args:
        text: Text to extract from
        
    Returns:
        Extracted JSON dict or None if extraction fails
    """
    match = JSON_PATTERN.search(text)
    if match:
        try:
            # Get the longest valid JSON string
            json_str = match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find a valid JSON substring
            try:
                # Find the outermost valid JSON object by checking for balanced braces
                open_braces = 0
                start_idx = text.find('{')
                
                if start_idx >= 0:
                    for i in range(start_idx, len(text)):
                        if text[i] == '{':
                            open_braces += 1
                        elif text[i] == '}':
                            open_braces -= 1
                            
                        # When we close the outermost brace, we have a complete JSON object
                        if open_braces == 0 and i > start_idx:
                            try:
                                return json.loads(text[start_idx:i+1])
                            except json.JSONDecodeError:
                                pass
            except:
                pass
    return None

def _extract_from_json_lines(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON by looking for complete JSON objects on individual lines.
    
    Args:
        text: Text to extract from
        
    Returns:
        Extracted JSON dict or None if extraction fails
    """
    # Look for JSON objects on individual lines
    match = JSON_LINES_PATTERN.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None

def _extract_json_comprehensive(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON using a comprehensive approach for difficult cases.
    
    This is a slower but more thorough method that scans for complex
    nested structures and handles common formatting issues.
    
    Args:
        text: Text to extract from
        
    Returns:
        Extracted JSON dict or None if extraction fails
    """
    # Clean up the text for better parsing
    # 1. Remove any leading/trailing text before first { and after last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx >= 0 and end_idx > start_idx:
        json_candidate = text[start_idx:end_idx+1]
        
        # 2. Fix common formatting issues
        # Replace single quotes with double quotes
        json_candidate = re.sub(r"'([^']*)'", r'"\1"', json_candidate)
        
        # Fix trailing commas before closing brackets
        json_candidate = re.sub(r',\s*}', '}', json_candidate)
        json_candidate = re.sub(r',\s*]', ']', json_candidate)
        
        # 3. Try to parse the JSON
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            # 4. If that fails, try a line-by-line reconstruction approach
            try:
                return _reconstruct_json(json_candidate)
            except:
                pass
    
    return None

def _reconstruct_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Reconstruct a valid JSON object from a malformed one.
    
    This function attempts to fix common JSON formatting issues by
    reconstructing the JSON structure line by line.
    
    Args:
        text: Potentially malformed JSON text
        
    Returns:
        Corrected JSON dict or None if reconstruction fails
    """
    # Split the text into lines
    lines = text.split('\n')
    
    # Remove comments and fix formatting issues
    cleaned_lines = []
    for line in lines:
        # Remove comments
        if '//' in line:
            line = line.split('//')[0]
        
        # Fix trailing commas and other issues
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    # Try to parse the cleaned JSON
    try:
        return json.loads('\n'.join(cleaned_lines))
    except json.JSONDecodeError:
        return None


def test_json_extraction():
    """Test the JSON extraction functions with various input formats."""
    test_cases = [
        # Clean JSON
        """{"name": "John", "age": 30}""",
        
        # JSON in code block
        """```json
        {
            "name": "John",
            "age": 30
        }
        ```""",
        
        # JSON with surrounding text
        """Here's the extracted information:
        {
            "name": "John",
            "age": 30
        }
        Let me know if you need anything else.""",
        
        # Malformed JSON with single quotes
        """Here's the data:
        {
            'name': 'John',
            'age': 30
        }""",
        
        # Malformed JSON with trailing commas
        """{
            "name": "John",
            "age": 30,
        }""",
        
        # Complex nested JSON with formatting issues
        """The information extracted:
        {
            "family_members": [
                {
                    "name": "John",
                    "age": 30,
                    "relation_to": [
                        {
                            "name": "Mary",
                            "relationship": "husband"
                        },
                    ]
                },
                {
                    "name": "Mary",
                    "age": 28,
                    "relation_to": [
                        {
                            "name": "John",
                            "relationship": "wife"
                        },
                    ]
                }
            ]
        }"""
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"Test case {i+1}:")
        print(f"Input: {test_case[:50]}{'...' if len(test_case) > 50 else ''}")
        result = extract_json_efficiently(test_case)
        print(f"Result: {result}")
        print(f"Success: {bool(result)}")
        print("")
    
    print("JSON extraction tests completed.")


if __name__ == "__main__":
    test_json_extraction()