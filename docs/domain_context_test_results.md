# Domain Context in Prompt Tests

## Overview
This document summarizes experiments testing whether Claude-3.7 models benefit from domain context provided before instructions in structured information extraction tasks. The tests were conducted using the sentence extraction evaluation framework.

## Experiment Design
Two prompt variations were tested:
1. **With Domain Context**: Added a paragraph explaining family relationships before the instruction
2. **Without Domain Context**: Only contained task instructions with no additional context

Both prompts contained identical instructions for extracting family member information and relationships from sentences.

### Prompt Templates
**With Domain Context:**
```
Family relationships are complex networks of biological, legal, and emotional connections that form the foundation of society. They involve parents, children, siblings, grandparents, aunts, uncles, and cousins, connected through birth, marriage, adoption, or other legal arrangements. These relationships often include important biographical details such as birth and death dates, occupations, education, medical conditions, and life achievements that define a person's life story and place in the family structure.

You will extract structured information about family members and relationships from a sentence...
```

**Without Domain Context:**
```
You will extract structured information about family members and relationships from a sentence...
```

### Test Data
The test used sentences from the existing evaluation datasets, including:
- Simple family relationship sentences
- Complex multi-generational family structures

Example test sentence:
```
My sister Jennifer was diagnosed with type 1 diabetes when she was just 7 years old in 1998, but that didn't stop her from becoming a marathon runner who completed the Boston Marathon in 2022.
```

## Results

### Quantitative Metrics
Both prompt variations showed identical performance across all key metrics:

| Metric | With Context | Without Context |
|--------|-------------|----------------|
| Value accuracy | 69.12% | 69.12% |
| Field extraction rate | 0.0% | 0.0% |
| Average sentence score | 55.9% | 55.9% |
| Success rate | 100.0% | 100.0% |

### Qualitative Analysis
Detailed examination of the extracted information revealed:
- Both prompts produced structurally identical outputs
- Both prompts made the exact same errors, including:
  - Relationship mismatches (e.g., "Mother" vs "mother")
  - Missing nested objects for complex family structures
  - Partial matches on achievement fields
- Adding domain context did not affect error patterns or successful extractions

## Conclusion
For structured information extraction tasks with detailed instructions, Claude-3.7 models do not show measurable benefits from adding domain context before task instructions. The model's performance, error patterns, and output structure were identical regardless of whether domain context was provided.

This suggests that for well-defined extraction tasks with comprehensive instructions, Claude-3.7 has sufficient understanding of the domain and task requirements without needing additional contextual information.

### Recommendations
- Focus on providing clear, detailed instructions rather than adding domain context for extraction tasks
- For complex extractions, use examples within the instruction set rather than general domain descriptions
- Consider that domain context may be more beneficial for open-ended or creative tasks than for structured extraction tasks

## Test Implementation Details
Tests were conducted using the extraction evaluation framework with the following configuration:
- Model: Claude-3.7 Sonnet
- Temperature: 0.0
- Multiple test sentences from family relationship datasets
- Cross-validation across sentence types

The test code and complete results are available in the repository.