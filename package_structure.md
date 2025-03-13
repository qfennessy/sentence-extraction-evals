# Sentence Extraction Package Structure

This document describes the reorganization of the sentence extraction evaluation framework into a proper Python package.

## Package Structure

The code has been reorganized into a well-structured Python package:

```
sentence_extraction/
├── __init__.py               # Main package exports
├── core/                     # Core extraction functionality
│   ├── __init__.py
│   └── extraction_eval.py    # Main extraction evaluation
├── optimizations/            # Performance optimizations
│   ├── __init__.py
│   ├── extraction_cache.py   # Response caching
│   ├── rate_limiting.py      # Adaptive rate limiting
│   └── parallel_processor.py # Parallel processing
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── metrics.py            # Metrics calculation
└── scripts/                  # Command-line tools
    ├── __init__.py
    └── extraction_eval.py    # CLI for extraction evaluation
```

## Key Changes

1. **Package Structure**: Created a proper Python package structure with separated modules.

2. **Core Functionality**: Moved the core extraction functionality into a dedicated module.

3. **Optimizations**: Segregated the optimization code into focused modules:
   - Response caching
   - Adaptive rate limiting
   - Parallel processing

4. **Command-Line Interface**: Created a standalone CLI script that can be installed as a console entry point.

5. **Setup Script**: Added a proper setup.py to make the package installable.

## Benefits

1. **Improved Maintainability**: Code is now organized into logical modules with clearly defined responsibilities.

2. **Better Reusability**: Components can be imported individually and reused across different projects.

3. **Simplified Installation**: The package can be installed directly with pip, either from PyPI or the repository.

4. **Command-Line Tool**: The package provides a convenient command-line interface through the console entry point.

5. **Cleaner Imports**: The organized structure allows for cleaner import statements.

## Usage Examples

### As a Library

```python
from sentence_extraction import evaluate_extraction, AnthropicAdapter, CLAUDE_MODELS

# Simple usage
results = evaluate_extraction(
    client, 
    prompt_template, 
    sentences, 
    CLAUDE_MODELS["sonnet"]
)

# With optimizations
from sentence_extraction.optimizations import ResponseCache

cache = ResponseCache()
results = evaluate_extraction(
    client, 
    prompt_template, 
    sentences, 
    model_id,
    parallel=True,
    max_workers=8
)
```

### As a Command-Line Tool

```bash
# After installing the package
extraction-eval --prompt-file prompt_template.txt --eval-data dataset.json --model sonnet
```

## Next Steps

1. **Documentation**: Add more comprehensive docstrings and documentation.

2. **Testing**: Create unit tests for each module.

3. **Additional Optimizations**: Implement Phase 2 optimizations as separate modules.

4. **Configuration System**: Add a comprehensive configuration system.