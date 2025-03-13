# sentence_extraction package
"""
Package for sentence extraction and evaluation.

This package provides tools for evaluating a language model's ability to extract 
structured information from unstructured text, with a focus on family relationships.

Main components:
- core: Core extraction functionality and evaluation
- optimizations: Performance optimizations for scaling extraction tasks
- utils: Utility functions for metrics and analysis
- scripts: Command-line tools for running evaluations

Usage:
    from sentence_extraction.core import evaluate_extraction, AnthropicAdapter
    from sentence_extraction.optimizations import ResponseCache
    from sentence_extraction.utils import calculate_metrics
"""

__version__ = "0.1.0"

# Export key functions at the package level
from .core.extraction_eval import (
    evaluate_extraction, 
    AnthropicAdapter, 
    OpenAIAdapter,
    GeminiAdapter,
    DeepSeekAdapter,
    load_data,
    format_prompt,
    extract_json_from_response,
    create_output_directory,
    save_results,
    CLAUDE_MODELS,
    OPENAI_MODELS,
    GEMINI_MODELS,
    DEEPSEEK_MODELS
)

from .optimizations import (
    ResponseCache,
    AdaptiveRateLimiter,
    ModelRateLimiter,
    ParallelProcessor,
    process_data_parallel
)

from .utils import calculate_metrics