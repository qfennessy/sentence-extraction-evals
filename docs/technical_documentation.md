# Sentence Extraction Framework Technical Documentation

## Overview

This document provides comprehensive technical documentation for the Family Information Extraction Evaluation Framework. The framework enables systematic evaluation of how well language models extract structured information from text, with a focus on family relationships and biographical data.

## Architecture

The framework follows a modular architecture:

```
sentence_extraction/
├── __init__.py               # Main package exports
├── core/                     # Core extraction functionality
│   ├── __init__.py
│   └── extraction_eval.py    # Main extraction evaluation
├── optimizations/            # Performance optimizations
│   ├── __init__.py
│   ├── extraction_cache.py   # Response caching
│   ├── json_extractor.py     # JSON parsing optimization
│   ├── optimized_metrics.py  # Fast metrics calculation
│   ├── parallel_processor.py # Parallel processing
│   ├── prompt_tier_system.py # Tiered prompt management
│   └── rate_limiting.py      # Adaptive rate limiting
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── metrics.py            # Metrics calculation
└── scripts/                  # Command-line tools
    ├── __init__.py
    └── extraction_eval.py    # CLI for extraction evaluation
```

## Core API Reference

### Model Adapters

#### Base Interface

```python
class ModelClient(Protocol):
    """Interface for model adapters."""
    
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """Call the model with the given prompt."""
        ...
```

#### Implementation Adapters

The framework supports multiple model providers through standardized adapters:

##### AnthropicAdapter

```python
class AnthropicAdapter:
    """Adapter for Anthropic Claude models."""
    
    def __init__(self, api_key: str):
        """Initialize the adapter with API key."""
        
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """Call the Anthropic Claude API."""
```

Models: `sonnet`, `opus`, `haiku`

##### OpenAIAdapter

```python
class OpenAIAdapter:
    """Adapter for OpenAI models."""
    
    def __init__(self, api_key: str):
        """Initialize the adapter with API key."""
        
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """Call the OpenAI API."""
```

Models: `gpt4o`, `gpt4`, `gpt35`

##### GeminiAdapter

```python
class GeminiAdapter:
    """Adapter for Google Gemini models."""
    
    def __init__(self, api_key: str):
        """Initialize the adapter with API key."""
        
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """Call the Google Gemini API."""
```

Models: `pro`, `pro-vision`, `ultra`

##### DeepSeekAdapter

```python
class DeepSeekAdapter:
    """Adapter for DeepSeek models."""
    
    def __init__(self, api_key: str):
        """Initialize the adapter with API key."""
        
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """Call the DeepSeek API."""
```

Models: `coder`, `chat`

### Extraction Evaluation Core

The core functionality handles prompt formatting, model interaction, and result processing:

```python
def load_data(prompt_file: str, eval_data_file: str, max_sentences: Optional[int] = None) -> Tuple[str, List[Dict]]:
    """Load prompt template and evaluation data."""

def format_prompt(prompt_template: str, sentence: str) -> str:
    """Format a prompt with a sentence."""

def extract_json_from_response(response: str) -> Dict:
    """Parse JSON from model response."""

def process_single_item(
    client: ModelClient, 
    prompt_template: str, 
    item: Dict, 
    model: str, 
    temperature: float = 0.0,
    rate_limiter: Optional["AdaptiveRateLimiter"] = None,
    cache: Optional["ResponseCache"] = None
) -> Dict:
    """Process a single sentence extraction."""

def evaluate_extraction(
    client: ModelClient,
    prompt_template: str,
    sentences: List[Dict],
    model: str,
    batch_size: int = 10,
    temperature: float = 0.0,
    parallel: bool = True,
    max_workers: int = 5
) -> List[Dict]:
    """Main extraction evaluation function."""
```

### Metrics Calculation

```python
def calculate_metrics(results: List[Dict], detailed: bool = True) -> Dict:
    """Calculate metrics from extraction results."""

def compare_extraction_results(expected: Dict, extracted: Dict) -> Dict:
    """Compare expected and extracted information."""

def create_metrics_summary(metrics: Dict) -> Dict:
    """Create a summary of metrics."""
```

## Optimization Components

### JSON Extraction

```python
def extract_json_efficiently(text: str) -> Dict:
    """Extract JSON using optimized pattern matching with fallbacks."""
```

Internal methods:
- `_extract_from_code_block(text)`
- `_extract_using_json_pattern(text)`
- `_extract_from_json_lines(text)`
- `_extract_json_comprehensive(text)`

### Response Caching

```python
class ResponseCache:
    """Cache for model responses to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = "./cache", expiration_hours: int = 72):
        """Initialize the cache with directory and expiration time."""
        
    def get(self, prompt: str, sentence: str, model: str) -> Optional[str]:
        """Get a cached response if available."""
        
    def set(self, prompt: str, sentence: str, model: str, response: str) -> None:
        """Cache a model response."""
        
    def clear_expired(self) -> int:
        """Clear expired cache entries."""
        
    def clear_all(self) -> int:
        """Clear all cache entries."""
        
    def get_stats(self) -> Dict:
        """Get cache statistics."""
```

### Parallel Processing

```python
class ParallelProcessor:
    """Parallel processing for batched operations."""
    
    def __init__(self, max_workers: int = 5, show_progress: bool = True):
        """Initialize with maximum workers and progress display option."""
        
    def process_batch(
        self, 
        items: List[Any], 
        process_func: Callable[[Any], Any],
        desc: str = "Processing"
    ) -> List[Any]:
        """Process a batch of items in parallel."""
        
    def get_stats(self) -> Dict:
        """Get processing statistics."""

def process_data_parallel(
    data_items: List[Any],
    process_func: Callable[[Any], Any],
    max_workers: int = 5,
    batch_size: int = 10,
    show_progress: bool = True
) -> List[Any]:
    """Convenience function for parallel processing with batching."""
```

### Optimized Metrics Calculation

```python
def normalize_structure_fast(data: Dict, sentence: str) -> Dict:
    """Normalize data structures efficiently."""

def deep_compare_optimized(
    expected: Dict,
    extracted: Dict,
    field_path: str = "",
    detailed_metrics: bool = True
) -> Dict:
    """Compare expected and extracted data with optimization."""

def compare_extraction_results(
    expected: Dict,
    extracted: Dict,
    fast_mode: bool = False
) -> Dict:
    """Compare extraction results with various options."""

def calculate_metrics_optimized(
    results: List[Dict],
    include_details: bool = True
) -> Dict:
    """Calculate comprehensive evaluation metrics optimized for speed."""
```

### Tiered Prompt System

```python
class TieredPromptSystem:
    """System for managing prompts of different complexity tiers."""
    
    def __init__(self, base_prompt_path: str, cache_dir: str = "./prompt_cache"):
        """Initialize with base prompt path and cache directory."""
        
    def get_prompt(self, tier: str) -> str:
        """Get prompt template for the specified tier."""
        
    def get_token_estimates(self) -> Dict[str, int]:
        """Get token usage estimates for each tier."""
        
    def get_recommended_tier(self, sentence: str) -> str:
        """Get recommended tier based on sentence complexity."""
        
    def analyze_sentence_complexity(self, sentence: str) -> Dict:
        """Analyze sentence complexity metrics."""
```

Tiers: 
- `"fast"`: Minimal prompt for simple sentences
- `"standard"`: Balanced prompt for typical sentences
- `"comprehensive"`: Detailed prompt for complex sentences

### Rate Limiting

```python
class AdaptiveRateLimiter:
    """Adaptive rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: float, burst_capacity: int = 5):
        """Initialize with rate limit and burst capacity."""
        
    def wait(self) -> float:
        """Wait until a request can be made and return wait time."""
        
    def get_stats(self) -> Dict:
        """Get rate limiting statistics."""
        
    def update_rate(self, new_requests_per_minute: float) -> None:
        """Update the rate limit."""

class ModelRateLimiter:
    """Rate limiter for multiple models."""
    
    def __init__(self, default_rates: Dict[str, float]):
        """Initialize with default rates for models."""
        
    def get_limiter(self, model: str) -> AdaptiveRateLimiter:
        """Get rate limiter for a specific model."""
        
    def wait(self, model: str) -> float:
        """Wait for rate limiting on the specified model."""
        
    def get_stats(self, model: Optional[str] = None) -> Dict:
        """Get rate limiting statistics for one or all models."""
```

## Command-Line Interface

The package provides a command-line interface through `extraction-eval`:

```
extraction-eval --model sonnet --prompt-file prompt_template.txt --eval-data eval_data.json
```

Full options:
- `--model`: Model to use (sonnet, opus, haiku, gpt4o, etc.)
- `--prompt-file`: Path to prompt template
- `--eval-data`: Path to evaluation data
- `--max-sentences`: Maximum sentences to evaluate
- `--temperature`: Temperature for model (0.0-1.0)
- `--parallel/--no-parallel`: Enable/disable parallel processing
- `--max-workers`: Number of parallel workers
- `--output-dir`: Output directory for results
- `--cache/--no-cache`: Enable/disable response caching
- `--provider`: Model provider override (anthropic, openai, google, deepseek)

## Advanced Features

### Fully Optimized Execution

```python
def run_fully_optimized_evaluation(
    prompt_file: str,
    eval_data: str,
    model: str,
    tier: str = "auto",
    parallel: bool = True,
    max_workers: int = 5,
    batch_size: int = 10,
    detailed_metrics: bool = True,
    cache_enabled: bool = True,
    rate_limiting: bool = True,
    provider: Optional[str] = None
) -> Dict:
    """Run evaluation with all optimizations enabled."""
```

### Cross-Validation Framework

```python
def run_cross_validation(
    datasets: List[str],
    models: List[str],
    prompt_variations: List[str],
    folds: int = 5,
    max_sentences: Optional[int] = None,
    output_dir: str = "cross_validation_results"
) -> Dict:
    """Run cross-validation across datasets, models, and prompts."""
```

### Prompt Refinement

```python
def refine_prompt(
    base_prompt: str,
    train_data: List[Dict],
    validation_data: List[Dict],
    model: str,
    max_iterations: int = 5
) -> Tuple[str, Dict]:
    """Refine a prompt through iterative improvement."""
```

## Data Formats

### Input Dataset Format

```json
{
  "sentences": [
    {
      "sentence": "My father David worked as a mechanic in Chicago for 35 years.",
      "extracted_information": {
        "name": "David",
        "relationship": "Father",
        "occupation": "mechanic",
        "location": "Chicago",
        "work_duration": "35 years"
      }
    }
  ]
}
```

### Complex Dataset Format

```json
{
  "sentences": [
    {
      "sentence": "My father David and my mother Sarah both grew up in Chicago, where dad worked as a mechanic for 35 years while mom was a teacher.",
      "extracted_information": {
        "family_members": [
          {
            "name": "David",
            "relationship": "Father",
            "occupation": "mechanic",
            "location": "Chicago",
            "work_duration": "35 years",
            "grew_up_in": "Chicago"
          },
          {
            "name": "Sarah",
            "relationship": "Mother",
            "occupation": "teacher",
            "location": "Chicago",
            "grew_up_in": "Chicago"
          }
        ]
      }
    }
  ]
}
```

### Evaluation Results Format

```json
{
  "model": "claude-3-sonnet-20250219",
  "prompt_file": "prompt_template.txt",
  "eval_data": "example_data.json",
  "runtime_seconds": 120.5,
  "timestamp": "2025-03-13T14:30:00",
  "results": [
    {
      "sentence": "My father David worked as a mechanic in Chicago for 35 years.",
      "expected": {
        "name": "David",
        "relationship": "Father",
        "occupation": "mechanic",
        "location": "Chicago",
        "work_duration": "35 years"
      },
      "extracted": {
        "name": "David",
        "relationship": "Father",
        "occupation": "mechanic",
        "location": "Chicago",
        "work_duration": "35 years"
      },
      "metrics": {
        "field_matches": 5,
        "field_errors": 0,
        "value_accuracy": 1.0
      }
    }
  ],
  "metrics": {
    "overall_value_accuracy": 0.95,
    "field_precision": 0.97,
    "field_recall": 0.93,
    "field_f1": 0.95,
    "relationship_accuracy": 0.98,
    "name_accuracy": 0.99
  }
}
```

## Environment Variables

- `ANTHROPIC_API_KEY`: API key for Anthropic Claude models
- `OPENAI_API_KEY`: API key for OpenAI models
- `GOOGLE_API_KEY`: API key for Google Gemini models
- `DEEPSEEK_API_KEY`: API key for DeepSeek models

## Development and Customization

### Custom Model Adapters

```python
class CustomModelAdapter:
    """Custom adapter for a new model provider."""
    
    def __init__(self, api_key: str):
        """Initialize the adapter."""
        self.api_key = api_key
        
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """Call the custom model API."""
        # Implementation details
        return response_text
```

### Custom Metrics

```python
def calculate_custom_metrics(results: List[Dict]) -> Dict:
    """Calculate custom evaluation metrics."""
    # Implementation details
    return custom_metrics
```

## Performance Considerations

- For datasets with >100 items, enable caching and parallel processing
- For benchmarking, use consistent temperature (0.0 recommended)
- The tiered prompt system provides significant efficiency gains
- Memory usage scales with batch size and number of parallel workers
- For large datasets, process in batches with `max_sentences` limit