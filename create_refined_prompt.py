#!/usr/bin/env python3
"""
Simple script to create a refined prompt focused on military information extraction
"""

import os
from pathlib import Path

# Create output directory
output_dir = Path("mini_refinement")
os.makedirs(output_dir, exist_ok=True)

# Load base prompt
with open("prompt_template.txt", 'r') as f:
    base_prompt = f.read()

# Create targeted refinements for military and temporal fields
refinements = [
    """## Enhanced extraction of military information

When extracting military service information, pay special attention to these fields:
- event (e.g., "World War II")
- time_period (e.g., "1942-1945")
- award (e.g., "Purple Heart")
- injury_location (e.g., "Normandy")

Example: In "My great-grandfather fought in World War II from 1942-1945 and received a Purple Heart after being wounded at Normandy."
- The "event" field should be: "World War II"
- The "time_period" field should be: "1942-1945"
- The "award" field should be: "Purple Heart"
- The "injury_location" field should be: "Normandy"
""",

    """## Handling complex temporal references

When extracting temporal information, ensure you properly identify:
- Specific time periods with start and end dates (e.g., "from 1942-1945")
- Duration of events or activities (e.g., "for 35 years")
- Sequential events with temporal markers (e.g., "before retiring in 2018")

Example: When a sentence like "worked in Chicago for 35 years before retiring in 2018" appears, capture both the duration ("35 years") and the significant event year ("2018").
"""
]

# Find insertion point in prompt (before the {SENTENCE} placeholder)
insertion_point = base_prompt.find("{SENTENCE}")
if insertion_point > 0:
    line_break = base_prompt.rfind("\n\n", 0, insertion_point)
    insertion_point = line_break if line_break > 0 else insertion_point

# Create refined prompt
refined_prompt = (
    base_prompt[:insertion_point] +
    "\n\n# ADDITIONAL GUIDANCE FOR SPECIFIC CASES\n\n" +
    "\n\n".join(refinements) +
    "\n\n" +
    base_prompt[insertion_point:]
)

# Save refined prompt
refined_prompt_path = output_dir / "refined_prompt.txt"
with open(refined_prompt_path, 'w') as f:
    f.write(refined_prompt)

print(f"Created refined prompt focused on military information extraction: {refined_prompt_path}")