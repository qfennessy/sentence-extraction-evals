# sentence_extraction.optimizations package
"""
Optimization modules for improving extraction performance.
"""

from .extraction_cache import ResponseCache
from .rate_limiting import AdaptiveRateLimiter, ModelRateLimiter
from .parallel_processor import ParallelProcessor, process_data_parallel