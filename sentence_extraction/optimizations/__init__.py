# sentence_extraction.optimizations package
"""
Optimization modules for improving extraction performance.

This package provides components for optimizing extraction speed and efficiency:

Phase 1 Optimizations:
- Parallel Processing: Process multiple requests concurrently
- Response Caching: Avoid redundant API calls
- Adaptive Rate Limiting: Optimize request rates

Phase 2 Optimizations:
- Tiered Prompt System: Different prompt tiers for performance/accuracy tradeoffs
- Optimized JSON Parsing: Faster and more robust extraction of JSON responses
- Metrics Calculation Optimization: Early termination and selective computation
"""

# Phase 1 Optimizations
from .extraction_cache import ResponseCache
from .rate_limiting import AdaptiveRateLimiter, ModelRateLimiter
from .parallel_processor import ParallelProcessor, process_data_parallel

# Phase 2 Optimizations
from .json_extractor import extract_json_efficiently
from .prompt_tier_system import TieredPromptSystem
from .optimized_metrics import (
    normalize_structure_fast,
    compare_extraction_results,
    calculate_metrics_optimized
)