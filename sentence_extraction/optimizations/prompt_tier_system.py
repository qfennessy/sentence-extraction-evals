#!/usr/bin/env python
"""
Tiered Prompt System for extraction optimization.

This module provides a tiered prompt system with three performance tiers:
- Fast: Minimal examples, core instructions only (~40% token reduction)
- Standard: Balanced examples and instructions (current approach)
- Comprehensive: Complete examples for complex cases

The system allows dynamically selecting the appropriate prompt tier based on 
task complexity, time constraints, or accuracy requirements.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

class TieredPromptSystem:
    """
    Manages a tiered system of prompts with different levels of detail and complexity.
    
    The system provides three tiers:
    - fast: Minimal examples, core instructions only (~40% token reduction)
    - standard: Balanced examples and instructions (current approach)
    - comprehensive: Complete examples for complex cases
    """
    
    def __init__(self, base_prompt_path: str, cache_dir: Optional[str] = None):
        """
        Initialize the tiered prompt system.
        
        Args:
            base_prompt_path: Path to the standard prompt template
            cache_dir: Directory to cache prompts (optional)
        """
        self.base_prompt_path = base_prompt_path
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Default paths for the different tiers
        self.prompt_paths = {
            "fast": self._get_tier_path("fast"),
            "standard": self._get_tier_path("standard"),
            "comprehensive": self._get_tier_path("comprehensive")
        }
        
        # Cache for loaded prompts
        self._prompt_cache = {}
        
        # Approximate token counts
        self._token_estimates = {
            "fast": 300,
            "standard": 700,
            "comprehensive": 1200
        }
        
        # Complexity indicators for auto-tier selection
        self.complexity_indicators = {
            "high": [
                r"adoptive|adopted|adoption",
                r"foster(ing|ed)?",
                r"step(mother|father|parent|child|son|daughter|brother|sister)",
                r"in-law",
                r"(maternal|paternal) (grand|great)",
                r"twin",
                r"diagnosed|diagnosis|medical condition",
                r"historical|history",
                r"\d{4}.*\d{4}",  # Multiple years mentioned
                r"graduated|earned a degree",
                r"married.*divorced",
                r"business|company|firm",
                r"army|military|navy|air force|marines",
                r"immigrated|emigrated|migration"
            ],
            "medium": [
                r"grand(mother|father|parent|child|son|daughter)",
                r"great([-\s]grand)?(mother|father|parent|child|son|daughter)",
                r"(aunt|uncle|cousin)",
                r"(brother|sister|sibling)",
                r"(husband|wife|spouse)",
                r"\b\d{4}\b",  # Year
                r"born|died|passed away",
                r"university|college|school",
                r"moved to|lived in",
                r"occupation|profession|job",
                r"retired|retirement"
            ]
        }
    
    def _get_tier_path(self, tier: str) -> str:
        """
        Get the file path for a specific prompt tier.
        
        Args:
            tier: The tier name ("fast", "standard", or "comprehensive")
            
        Returns:
            The file path for the prompt tier
        """
        # Get the base directory and file name
        base_dir = os.path.dirname(self.base_prompt_path)
        base_name = os.path.basename(self.base_prompt_path)
        
        # If the base path is a standard prompt, use it directly
        if f"_template_{tier}.txt" in base_name:
            return self.base_prompt_path
        
        # Check if it's in the optimizations/prompts directory
        prompts_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "optimizations", 
            "prompts"
        )
        
        # For standard prompt, use the base path
        if tier == "standard":
            # If we have a standard version in the prompts dir, use that
            std_path = os.path.join(prompts_dir, f"prompt_template_standard.txt")
            if os.path.exists(std_path):
                return std_path
            # Otherwise use the base path
            return self.base_prompt_path
            
        # For other tiers, construct the path in the prompts directory
        tier_path = os.path.join(prompts_dir, f"prompt_template_{tier}.txt")
        
        # If the tier path exists, use it
        if os.path.exists(tier_path):
            return tier_path
        
        # Otherwise, use the standard path for now
        return self.base_prompt_path
    
    def get_prompt(self, tier: str = "standard") -> str:
        """
        Get the prompt for a specific tier.
        
        Args:
            tier: The tier to use ("fast", "standard", or "comprehensive")
            
        Returns:
            The prompt text for the specified tier
        """
        # Normalize tier name
        tier = tier.lower()
        if tier not in ["fast", "standard", "comprehensive"]:
            tier = "standard"
        
        # Check cache first
        if tier in self._prompt_cache:
            return self._prompt_cache[tier]
        
        # Get file path for the tier
        prompt_path = self.prompt_paths.get(tier, self.prompt_paths["standard"])
        
        # Load the prompt
        try:
            with open(prompt_path, 'r') as f:
                prompt = f.read().strip()
        except FileNotFoundError:
            # Fall back to standard if the tier file is not found
            with open(self.prompt_paths["standard"], 'r') as f:
                prompt = f.read().strip()
        
        # Cache the prompt
        self._prompt_cache[tier] = prompt
        
        return prompt
    
    def get_token_estimates(self) -> Dict[str, int]:
        """
        Get approximate token counts for each tier.
        
        Returns:
            Dictionary mapping tier names to token counts
        """
        return self._token_estimates.copy()
    
    def get_recommended_tier(self, sentence: str) -> str:
        """
        Recommend the appropriate tier based on sentence complexity.
        
        Args:
            sentence: The input sentence
            
        Returns:
            The recommended tier ("fast", "standard", or "comprehensive")
        """
        # Check for high complexity indicators
        for pattern in self.complexity_indicators["high"]:
            if re.search(pattern, sentence, re.IGNORECASE):
                return "comprehensive"
        
        # Check for medium complexity indicators
        medium_count = 0
        for pattern in self.complexity_indicators["medium"]:
            if re.search(pattern, sentence, re.IGNORECASE):
                medium_count += 1
        
        # If multiple medium complexity indicators, use comprehensive
        if medium_count >= 3:
            return "comprehensive"
        # If some medium complexity, use standard
        elif medium_count >= 1:
            return "standard"
        # Otherwise use fast
        else:
            return "fast"
    
    def analyze_sentence_complexity(self, sentence: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a sentence and provide detailed reasoning.
        
        Args:
            sentence: The input sentence
            
        Returns:
            Dictionary with complexity analysis
        """
        high_matches = []
        medium_matches = []
        
        # Check for high complexity indicators
        for pattern in self.complexity_indicators["high"]:
            if re.search(pattern, sentence, re.IGNORECASE):
                high_matches.append(pattern)
        
        # Check for medium complexity indicators
        for pattern in self.complexity_indicators["medium"]:
            if re.search(pattern, sentence, re.IGNORECASE):
                medium_matches.append(pattern)
        
        # Determine overall complexity
        if high_matches:
            complexity = "high"
            recommended_tier = "comprehensive"
        elif len(medium_matches) >= 3:
            complexity = "high"
            recommended_tier = "comprehensive"
        elif medium_matches:
            complexity = "medium"
            recommended_tier = "standard"
        else:
            complexity = "low"
            recommended_tier = "fast"
        
        return {
            "sentence": sentence,
            "complexity": complexity,
            "recommended_tier": recommended_tier,
            "high_complexity_matches": high_matches,
            "medium_complexity_matches": medium_matches,
            "match_count": {
                "high": len(high_matches),
                "medium": len(medium_matches)
            }
        }


def test_tiered_prompt_system():
    """Test the tiered prompt system."""
    # Sample sentences of varying complexity
    sentences = [
        "My father works as a teacher.",
        "My sister Jennifer got married last year and moved to Boston.",
        "My paternal grandparents, Harold and Margaret, owned a farm for 47 years.",
        "My adopted brother Miguel came from Guatemala in 2008 when he was 6 years old."
    ]
    
    # Find the default prompt template
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    prompt_path = os.path.join(base_dir, "prompt_template.txt")
    
    # Create the tiered prompt system
    prompt_system = TieredPromptSystem(prompt_path)
    
    # Test the tier selection
    print("Testing tier selection:")
    for sentence in sentences:
        analysis = prompt_system.analyze_sentence_complexity(sentence)
        print(f"Sentence: {sentence}")
        print(f"Complexity: {analysis['complexity']}")
        print(f"Recommended tier: {analysis['recommended_tier']}")
        print(f"High complexity matches: {analysis['high_complexity_matches']}")
        print(f"Medium complexity matches: {analysis['medium_complexity_matches']}")
        print()
    
    # Test prompt loading
    print("Testing prompt loading:")
    for tier in ["fast", "standard", "comprehensive"]:
        prompt = prompt_system.get_prompt(tier)
        token_estimate = prompt_system.get_token_estimates()[tier]
        print(f"Loaded {tier} prompt: {len(prompt)} characters, ~{token_estimate} tokens")
    
    print("Tiered prompt system tests completed successfully.")


if __name__ == "__main__":
    test_tiered_prompt_system()