#!/usr/bin/env python
"""
Optimized JSON extraction module for efficient parsing of model responses.
Uses compiled regex patterns and optimized algorithms for faster extraction.
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple


# Pre-compile regex patterns for better performance
JSON_PATTERN = re.compile(r'({[\s\S]*?})(?=\s*$|\s*[,\]}]|\s*$)')
JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*([\s\S]*?)\s*```')
JSON_ARRAY_PATTERN = re.compile(r'(\[[\s\S]*?\])(?=\s*$|\s*[,\]}]|\s*$)')


def extract_json_efficiently(text: str) -> Dict[str, Any]:
    """
    Extract JSON from text using optimized pattern matching algorithms.
    
    Tries multiple extraction strategies in order of decreasing performance:
    1. Direct regex pattern match (fastest)
    2. Code block extraction (common in markdown responses)
    3. Line-by-line reconstruction (most robust, but slower)
    
    Args:
        text: The model response text to extract JSON from
        
    Returns:
        Extracted JSON as a dictionary, or empty dict if extraction fails
    """
    if not text:
        return {}
    
    # Try direct pattern match first (fastest method)
    result = _extract_via_regex(text)
    if result:
        return result
    
    # Try extracting from code blocks (common in markdown responses)
    result = _extract_from_code_block(text)
    if result:
        return result
    
    # Try array pattern for list responses
    result = _extract_array(text)
    if result and isinstance(result, list):
        # Convert array to dict if possible
        try:
            return {"results": result}
        except (TypeError, ValueError):
            pass
    
    # Fall back to line-by-line extraction (most robust but slower)
    return _extract_line_by_line(text)


def _extract_via_regex(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON using regex pattern matching."""
    # First try to find a complete JSON object
    match = JSON_PATTERN.search(text)
    if match:
        try:
            json_str = match.group(1)
            # If there are multiple matches, pick the largest one (likely the full result)
            if len(match.groups()) > 1:
                json_candidates = [g for g in match.groups() if g]
                json_str = max(json_candidates, key=len)
            
            # Try loading the extracted JSON
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If it fails, continue to other methods
            pass
    
    return None


def _extract_from_code_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from markdown code blocks."""
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        try:
            json_str = match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If it fails, continue to other methods
            pass
    
    return None


def _extract_array(text: str) -> Optional[List[Any]]:
    """Extract JSON array using regex pattern matching."""
    match = JSON_ARRAY_PATTERN.search(text)
    if match:
        try:
            json_str = match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If it fails, continue to other methods
            pass
    
    return None


def _extract_line_by_line(text: str) -> Dict[str, Any]:
    """
    Extract JSON by combining lines that appear to be part of JSON.
    This is a more robust method but slower than pattern matching.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Look for start of JSON object or array
    json_lines = []
    recording = False
    brace_count = 0
    bracket_count = 0
    
    for line in lines:
        # Start recording when we see opening brace or bracket
        if not recording and ('{' in line or '[' in line):
            recording = True
        
        if recording:
            json_lines.append(line)
            
            # Count braces and brackets to track nesting
            brace_count += line.count('{') - line.count('}')
            bracket_count += line.count('[') - line.count(']')
            
            # If all braces and brackets are balanced, we might have a complete JSON
            if brace_count <= 0 and bracket_count <= 0:
                # Try to parse what we have
                try:
                    json_str = ' '.join(json_lines)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Continue looking if this isn't valid JSON
                    pass
    
    # If we collected lines but didn't get a valid JSON, try one more time with everything
    if json_lines:
        try:
            json_str = ' '.join(json_lines)
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Last attempt: try to find and fix common JSON errors
            fixed_json = _attempt_json_repair(json_str)
            if fixed_json:
                return fixed_json
    
    # If all strategies fail, return empty dict
    return {}


def _attempt_json_repair(json_str: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to repair common JSON formatting errors.
    
    Args:
        json_str: The potentially malformed JSON string
        
    Returns:
        Repaired JSON dictionary or None if repair failed
    """
    try:
        # Common error: Trailing comma in objects/arrays
        json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
        
        # Common error: Missing quotes around keys
        json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1 "\2":', json_str)
        
        # Common error: Single quotes instead of double quotes
        if "'" in json_str and '"' not in json_str:
            json_str = json_str.replace("'", '"')
        
        # Try to parse the repaired JSON
        return json.loads(json_str)
    except (json.JSONDecodeError, Exception):
        return None


# Analysis mode for debugging extraction issues
def analyze_extraction(text: str) -> Dict[str, Any]:
    """
    Analyze a response to diagnose JSON extraction issues.
    
    Args:
        text: The model response text
        
    Returns:
        Dictionary with analysis results and extraction attempts
    """
    analysis = {
        "text_length": len(text),
        "has_json_pattern": bool(JSON_PATTERN.search(text)),
        "has_code_block": bool(JSON_BLOCK_PATTERN.search(text)),
        "has_json_array": bool(JSON_ARRAY_PATTERN.search(text)),
        "extraction_results": [],
        "final_result": {}
    }
    
    # Try each extraction method and record results
    methods = [
        ("regex", _extract_via_regex),
        ("code_block", _extract_from_code_block),
        ("array", _extract_array),
        ("line_by_line", _extract_line_by_line)
    ]
    
    for name, method in methods:
        try:
            start_time = json._default_encoder.default(datetime.datetime.now())
            result = method(text)
            end_time = json._default_encoder.default(datetime.datetime.now())
            
            analysis["extraction_results"].append({
                "method": name,
                "success": result is not None,
                "result_type": type(result).__name__ if result is not None else None,
                "result_length": len(json.dumps(result)) if result is not None else 0,
                "timing": {
                    "start": start_time,
                    "end": end_time
                }
            })
            
            # Save first successful result as the final one
            if result and not analysis["final_result"]:
                analysis["final_result"] = result
                
        except Exception as e:
            analysis["extraction_results"].append({
                "method": name,
                "success": False,
                "error": str(e)
            })
    
    return analysis


# Test function
def test_extractor():
    """Test the JSON extractor with various input types."""
    test_cases = [
        # Clean JSON object
        '{"name": "John", "age": 30, "city": "New York"}',
        
        # JSON in markdown code block
        '```json\n{"name": "John", "age": 30, "city": "New York"}\n```',
        
        # JSON with extra text around it
        'Here is the result: {"name": "John", "age": 30, "city": "New York"} as requested.',
        
        # Malformed JSON with common errors
        '{name: "John", "age": 30, "city": "New York",}',
        
        # JSON with single quotes
        "{'name': 'John', 'age': 30, 'city': 'New York'}",
        
        # JSON array
        '[{"name": "John", "age": 30}, {"name": "Jane", "age": 28}]',
        
        # Multi-line JSON with whitespace
        """
        {
            "name": "John",
            "age": 30,
            "city": "New York"
        }
        """
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print("-" * 40)
        print(f"Input: {test_case[:50]}..." if len(test_case) > 50 else f"Input: {test_case}")
        
        result = extract_json_efficiently(test_case)
        print(f"Result: {result}")
        
        # Verify if extraction was successful
        if result:
            print("✓ Extraction successful")
        else:
            print("✗ Extraction failed")
    
    print("\nExtractor tests completed!")


if __name__ == "__main__":
    import datetime  # For timing in analysis mode
    test_extractor()