# Extraction Failure Analysis

## Summary

- Total results analyzed: 3
- Failures: 1 (33.3%)
- Successes: 2

## Problematic Fields

### relationship

- Overall error rate: 100.0%
- Missing: 1 (100.0%)
- Incorrect: 0 (0.0%)
- Total occurrences: 1

### event

- Overall error rate: 100.0%
- Missing: 1 (100.0%)
- Incorrect: 0 (0.0%)
- Total occurrences: 1

### time_period

- Overall error rate: 100.0%
- Missing: 1 (100.0%)
- Incorrect: 0 (0.0%)
- Total occurrences: 1

### award

- Overall error rate: 100.0%
- Missing: 1 (100.0%)
- Incorrect: 0 (0.0%)
- Total occurrences: 1

### injury_location

- Overall error rate: 100.0%
- Missing: 1 (100.0%)
- Incorrect: 0 (0.0%)
- Total occurrences: 1

## Common Patterns in Failures

### Complex Temporal

Found 1 examples, including:

- "My great-grandfather fought in World War II from 1942-1945 and received a Purple Heart after being wounded at Normandy."

## Sentence Complexity Comparison

- Failed sentences avg length: 19.0 words (vs. 30.0 for successes)
- Failed sentences avg commas: 0.0 (vs. 3.0 for successes)

## Prompt Improvement Recommendations

### 1. Improve extraction of 'relationship' field

**Evidence:** 1 missing and 0 incorrect out of 1 occurrences

**Suggestion:** Add explicit examples showing correct extraction of this field


### 2. Improve extraction of 'event' field

**Evidence:** 1 missing and 0 incorrect out of 1 occurrences

**Suggestion:** Add explicit examples showing correct extraction of this field


### 3. Improve extraction of 'time_period' field

**Evidence:** 1 missing and 0 incorrect out of 1 occurrences

**Suggestion:** Add explicit examples showing correct extraction of this field


### 4. Improve extraction of 'award' field

**Evidence:** 1 missing and 0 incorrect out of 1 occurrences

**Suggestion:** Add explicit examples showing correct extraction of this field


### 5. Improve extraction of 'injury_location' field

**Evidence:** 1 missing and 0 incorrect out of 1 occurrences

**Suggestion:** Add explicit examples showing correct extraction of this field


### 6. Improve extraction of rare field: 'time_period'

**Evidence:** This field appears only in failure cases

**Suggestion:** Add specific examples with this field to the prompt


### 7. Improve extraction of rare field: 'event'

**Evidence:** This field appears only in failure cases

**Suggestion:** Add specific examples with this field to the prompt


### 8. Improve extraction of rare field: 'injury_location'

**Evidence:** This field appears only in failure cases

**Suggestion:** Add specific examples with this field to the prompt


### 9. Improve extraction of rare field: 'award'

**Evidence:** This field appears only in failure cases

**Suggestion:** Add specific examples with this field to the prompt


