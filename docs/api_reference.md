# Sentence Extraction Framework API Reference

## Core Modules

### Model Adapters

```python
from sentence_extraction import AnthropicAdapter, OpenAIAdapter, GeminiAdapter, DeepSeekAdapter
```

#### AnthropicAdapter

```python
class AnthropicAdapter:
    """
    Adapter for Anthropic Claude models.
    
    Attributes:
        api_key (str): Anthropic API key
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the adapter with API key.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY environment variable.
        """
    
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """
        Call the Anthropic Claude API.
        
        Args:
            prompt: The prompt to send to the API
            model: Model identifier (sonnet, opus, haiku)
            temperature: Temperature for sampling (0.0-1.0)
            
        Returns:
            Model response as string
        """
```

#### OpenAIAdapter

```python
class OpenAIAdapter:
    """
    Adapter for OpenAI models.
    
    Attributes:
        api_key (str): OpenAI API key
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the adapter with API key.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
        """
    
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """
        Call the OpenAI API.
        
        Args:
            prompt: The prompt to send to the API
            model: Model identifier (gpt4o, gpt4, gpt35)
            temperature: Temperature for sampling (0.0-1.0)
            
        Returns:
            Model response as string
        """
```

#### GeminiAdapter

```python
class GeminiAdapter:
    """
    Adapter for Google Gemini models.
    
    Attributes:
        api_key (str): Google API key
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the adapter with API key.
        
        Args:
            api_key: Google API key. If None, reads from GOOGLE_API_KEY environment variable.
        """
    
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """
        Call the Google Gemini API.
        
        Args:
            prompt: The prompt to send to the API
            model: Model identifier (pro, pro-vision, ultra)
            temperature: Temperature for sampling (0.0-1.0)
            
        Returns:
            Model response as string
        """
```

#### DeepSeekAdapter

```python
class DeepSeekAdapter:
    """
    Adapter for DeepSeek models.
    
    Attributes:
        api_key (str): DeepSeek API key
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the adapter with API key.
        
        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY environment variable.
        """
    
    def __call__(self, prompt: str, model: str, temperature: float = 0.0) -> str:
        """
        Call the DeepSeek API.
        
        Args:
            prompt: The prompt to send to the API
            model: Model identifier (coder, chat)
            temperature: Temperature for sampling (0.0-1.0)
            
        Returns:
            Model response as string
        """
```

### Data Loading and Processing

```python
from sentence_extraction import load_data, format_prompt, extract_json_from_response
```

#### load_data

```python
def load_data(
    prompt_file: str, 
    eval_data_file: str, 
    max_sentences: Optional[int] = None
) -> Tuple[str, List[Dict]]:
    """
    Load prompt template and evaluation data.
    
    Args:
        prompt_file: Path to prompt template file
        eval_data_file: Path to evaluation data JSON file
        max_sentences: Maximum number of sentences to load (None for all)
        
    Returns:
        Tuple of (prompt_template, sentences)
        
    Raises:
        FileNotFoundError: If files do not exist
        json.JSONDecodeError: If evaluation data is not valid JSON
    """
```

#### format_prompt

```python
def format_prompt(prompt_template: str, sentence: str) -> str:
    """
    Format a prompt with a sentence.
    
    Args:
        prompt_template: Prompt template with {SENTENCE} placeholder
        sentence: Sentence to insert into template
        
    Returns:
        Formatted prompt
    """
```

#### extract_json_from_response

```python
def extract_json_from_response(response: str) -> Dict:
    """
    Parse JSON from model response.
    
    Args:
        response: Model response text
        
    Returns:
        Extracted JSON as dictionary
        
    Raises:
        ValueError: If no valid JSON could be extracted
    """
```

### Core Evaluation

```python
from sentence_extraction import evaluate_extraction, process_single_item
```

#### process_single_item

```python
def process_single_item(
    client: ModelClient, 
    prompt_template: str, 
    item: Dict, 
    model: str, 
    temperature: float = 0.0,
    rate_limiter: Optional["AdaptiveRateLimiter"] = None,
    cache: Optional["ResponseCache"] = None
) -> Dict:
    """
    Process a single sentence extraction.
    
    Args:
        client: Model adapter client
        prompt_template: Prompt template with {SENTENCE} placeholder
        item: Dictionary with sentence and extracted_information
        model: Model identifier
        temperature: Temperature for sampling
        rate_limiter: Optional rate limiter
        cache: Optional response cache
        
    Returns:
        Dictionary with processing results
    """
```

#### evaluate_extraction

```python
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
    """
    Main extraction evaluation function.
    
    Args:
        client: Model adapter client
        prompt_template: Prompt template with {SENTENCE} placeholder
        sentences: List of sentence dictionaries
        model: Model identifier
        batch_size: Processing batch size
        temperature: Temperature for sampling
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of extraction results
    """
```

### Metrics Calculation

```python
from sentence_extraction import calculate_metrics, compare_extraction_results
```

#### calculate_metrics

```python
def calculate_metrics(results: List[Dict], detailed: bool = True) -> Dict:
    """
    Calculate metrics from extraction results.
    
    Args:
        results: List of extraction results
        detailed: Whether to include detailed metrics
        
    Returns:
        Dictionary of metrics
    """
```

#### compare_extraction_results

```python
def compare_extraction_results(expected: Dict, extracted: Dict) -> Dict:
    """
    Compare expected and extracted information.
    
    Args:
        expected: Expected extraction result
        extracted: Actual extraction result
        
    Returns:
        Comparison metrics dictionary
    """
```

## Optimization Modules

### Response Caching

```python
from sentence_extraction.optimizations import ResponseCache
```

```python
class ResponseCache:
    """
    Cache for model responses to avoid redundant API calls.
    
    Attributes:
        cache_dir (str): Directory for storing cache files
        expiration_hours (int): Cache entry expiration time in hours
    """
    
    def __init__(self, cache_dir: str = "./cache", expiration_hours: int = 72):
        """
        Initialize the cache with directory and expiration time.
        
        Args:
            cache_dir: Directory for storing cache files
            expiration_hours: Cache entry expiration time in hours
        """
    
    def get(self, prompt: str, sentence: str, model: str) -> Optional[str]:
        """
        Get a cached response if available.
        
        Args:
            prompt: Prompt used to generate response
            sentence: Input sentence
            model: Model identifier
            
        Returns:
            Cached response or None if not found
        """
    
    def set(self, prompt: str, sentence: str, model: str, response: str) -> None:
        """
        Cache a model response.
        
        Args:
            prompt: Prompt used to generate response
            sentence: Input sentence
            model: Model identifier
            response: Model response to cache
        """
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of entries cleared
        """
    
    def clear_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
```

### Parallel Processing

```python
from sentence_extraction.optimizations import ParallelProcessor, process_data_parallel
```

```python
class ParallelProcessor:
    """
    Parallel processing for batched operations.
    
    Attributes:
        max_workers (int): Maximum number of parallel workers
        show_progress (bool): Whether to show progress bar
    """
    
    def __init__(self, max_workers: int = 5, show_progress: bool = True):
        """
        Initialize with maximum workers and progress display option.
        
        Args:
            max_workers: Maximum number of parallel workers
            show_progress: Whether to show progress bar
        """
    
    def process_batch(
        self, 
        items: List[Any], 
        process_func: Callable[[Any], Any],
        desc: str = "Processing"
    ) -> List[Any]:
        """
        Process a batch of items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            desc: Description for progress bar
            
        Returns:
            List of processed items
        """
    
    def get_stats(self) -> Dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
```

```python
def process_data_parallel(
    data_items: List[Any],
    process_func: Callable[[Any], Any],
    max_workers: int = 5,
    batch_size: int = 10,
    show_progress: bool = True
) -> List[Any]:
    """
    Convenience function for parallel processing with batching.
    
    Args:
        data_items: List of items to process
        process_func: Function to apply to each item
        max_workers: Maximum number of parallel workers
        batch_size: Size of batches for processing
        show_progress: Whether to show progress bar
        
    Returns:
        List of processed items
    """
```

### JSON Extraction

```python
from sentence_extraction.optimizations import extract_json_efficiently
```

```python
def extract_json_efficiently(text: str) -> Dict:
    """
    Extract JSON using optimized pattern matching with fallbacks.
    
    Attempts multiple extraction methods in order of reliability and performance.
    
    Args:
        text: Text containing JSON
        
    Returns:
        Extracted JSON as dictionary
        
    Raises:
        ValueError: If no valid JSON could be extracted
    """
```

### Tiered Prompt System

```python
from sentence_extraction.optimizations import TieredPromptSystem
```

```python
class TieredPromptSystem:
    """
    System for managing prompts of different complexity tiers.
    
    Attributes:
        base_prompt_path (str): Path to base prompt template
        cache_dir (str): Directory for prompt cache
    """
    
    def __init__(self, base_prompt_path: str, cache_dir: str = "./prompt_cache"):
        """
        Initialize with base prompt path and cache directory.
        
        Args:
            base_prompt_path: Path to base prompt template
            cache_dir: Directory for prompt cache
        """
    
    def get_prompt(self, tier: str) -> str:
        """
        Get prompt template for the specified tier.
        
        Args:
            tier: Prompt tier (fast, standard, comprehensive)
            
        Returns:
            Prompt template for specified tier
            
        Raises:
            ValueError: If tier is not valid
        """
    
    def get_token_estimates(self) -> Dict[str, int]:
        """
        Get token usage estimates for each tier.
        
        Returns:
            Dictionary mapping tier to estimated token count
        """
    
    def get_recommended_tier(self, sentence: str) -> str:
        """
        Get recommended tier based on sentence complexity.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Recommended tier (fast, standard, comprehensive)
        """
    
    def analyze_sentence_complexity(self, sentence: str) -> Dict:
        """
        Analyze sentence complexity metrics.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Dictionary with complexity metrics
        """
```

### Rate Limiting

```python
from sentence_extraction.optimizations import AdaptiveRateLimiter, ModelRateLimiter
```

```python
class AdaptiveRateLimiter:
    """
    Adaptive rate limiter for API requests.
    
    Attributes:
        requests_per_minute (float): Rate limit
        burst_capacity (int): Number of requests allowed in a burst
    """
    
    def __init__(self, requests_per_minute: float, burst_capacity: int = 5):
        """
        Initialize with rate limit and burst capacity.
        
        Args:
            requests_per_minute: Rate limit in requests per minute
            burst_capacity: Number of requests allowed in a burst
        """
    
    def wait(self) -> float:
        """
        Wait until a request can be made and return wait time.
        
        Returns:
            Wait time in seconds
        """
    
    def get_stats(self) -> Dict:
        """
        Get rate limiting statistics.
        
        Returns:
            Dictionary with rate limiting statistics
        """
    
    def update_rate(self, new_requests_per_minute: float) -> None:
        """
        Update the rate limit.
        
        Args:
            new_requests_per_minute: New rate limit in requests per minute
        """
```

```python
class ModelRateLimiter:
    """
    Rate limiter for multiple models.
    
    Attributes:
        default_rates (Dict[str, float]): Default rate limits by model
    """
    
    def __init__(self, default_rates: Dict[str, float]):
        """
        Initialize with default rates for models.
        
        Args:
            default_rates: Dictionary mapping model to rate limit
        """
    
    def get_limiter(self, model: str) -> AdaptiveRateLimiter:
        """
        Get rate limiter for a specific model.
        
        Args:
            model: Model identifier
            
        Returns:
            AdaptiveRateLimiter for the model
        """
    
    def wait(self, model: str) -> float:
        """
        Wait for rate limiting on the specified model.
        
        Args:
            model: Model identifier
            
        Returns:
            Wait time in seconds
        """
    
    def get_stats(self, model: Optional[str] = None) -> Dict:
        """
        Get rate limiting statistics for one or all models.
        
        Args:
            model: Model identifier or None for all models
            
        Returns:
            Dictionary with rate limiting statistics
        """
```

### Optimized Metrics

```python
from sentence_extraction.optimizations import calculate_metrics_optimized
```

```python
def calculate_metrics_optimized(
    results: List[Dict],
    include_details: bool = True
) -> Dict:
    """
    Calculate comprehensive evaluation metrics optimized for speed.
    
    Args:
        results: List of extraction results
        include_details: Whether to include detailed metrics
        
    Returns:
        Dictionary of metrics
    """
```

## Advanced Features

### Cross-Validation

```python
from sentence_extraction import run_cross_validation
```

```python
def run_cross_validation(
    datasets: List[str],
    models: List[str],
    prompt_variations: List[str],
    folds: int = 5,
    max_sentences: Optional[int] = None,
    output_dir: str = "cross_validation_results",
    parallel: bool = True,
    max_workers: int = 5
) -> Dict:
    """
    Run cross-validation across datasets, models, and prompts.
    
    Args:
        datasets: List of dataset file paths
        models: List of model identifiers
        prompt_variations: List of prompt template file paths
        folds: Number of folds for cross-validation
        max_sentences: Maximum sentences per fold
        output_dir: Directory for output
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        
    Returns:
        Cross-validation results dictionary
    """
```

### Prompt Refinement

```python
from sentence_extraction import refine_prompt
```

```python
def refine_prompt(
    base_prompt: str,
    train_data: List[Dict],
    validation_data: List[Dict],
    model: str,
    max_iterations: int = 5,
    client: Optional[ModelClient] = None
) -> Tuple[str, Dict]:
    """
    Refine a prompt through iterative improvement.
    
    Args:
        base_prompt: Initial prompt template
        train_data: Training data
        validation_data: Validation data
        model: Model identifier
        max_iterations: Maximum refinement iterations
        client: Model adapter client (creates one if None)
        
    Returns:
        Tuple of (refined_prompt, performance_metrics)
    """
```

### Dataset Splitting

```python
from sentence_extraction import split_dataset
```

```python
def split_dataset(
    dataset_path: str,
    output_dir: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Split a dataset into train, validation, and test sets.
    
    Args:
        dataset_path: Path to dataset file
        output_dir: Directory for output
        train_ratio: Ratio of data for training
        validation_ratio: Ratio of data for validation
        test_ratio: Ratio of data for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping split names to file paths
    """
```

### Fully Optimized Evaluation

```python
from sentence_extraction import run_fully_optimized_evaluation
```

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
    provider: Optional[str] = None,
    max_sentences: Optional[int] = None,
    temperature: float = 0.0
) -> Dict:
    """
    Run evaluation with all optimizations enabled.
    
    Args:
        prompt_file: Path to prompt template
        eval_data: Path to evaluation data
        model: Model identifier
        tier: Prompt tier (fast, standard, comprehensive, auto)
        parallel: Whether to use parallel processing
        max_workers: Maximum number of parallel workers
        batch_size: Processing batch size
        detailed_metrics: Whether to include detailed metrics
        cache_enabled: Whether to enable response caching
        rate_limiting: Whether to enable rate limiting
        provider: Model provider override
        max_sentences: Maximum sentences to evaluate
        temperature: Temperature for sampling
        
    Returns:
        Evaluation results dictionary
    """
```

## Utility Functions

### Output Management

```python
from sentence_extraction import create_output_directory, save_results
```

```python
def create_output_directory(
    model_name: str, 
    prompt_file_path: str, 
    provider: str = "claude"
) -> str:
    """
    Create timestamped output directory.
    
    Args:
        model_name: Model identifier
        prompt_file_path: Path to prompt template
        provider: Model provider
        
    Returns:
        Path to created directory
    """
```

```python
def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: str,
    model_name: str,
    eval_data_path: str,
    runtime_seconds: float,
    prompt_file_path: str,
    model_provider: str = "claude"
) -> None:
    """
    Save evaluation results to files.
    
    Args:
        results: List of extraction results
        metrics: Dictionary of metrics
        output_dir: Directory for output
        model_name: Model identifier
        eval_data_path: Path to evaluation data
        runtime_seconds: Runtime in seconds
        prompt_file_path: Path to prompt template
        model_provider: Model provider
    """
```

### Constants

```python
from sentence_extraction import (
    CLAUDE_MODELS, 
    OPENAI_MODELS,
    GEMINI_MODELS,
    DEEPSEEK_MODELS,
    DEFAULT_RATE_LIMITS
)
```

```python
# Model mappings
CLAUDE_MODELS: Dict[str, str] = {
    "sonnet": "claude-3-7-sonnet-20250219",
    "opus": "claude-3-7-opus-20250301",
    "haiku": "claude-3-5-haiku-20240307"
}

OPENAI_MODELS: Dict[str, str] = {
    "gpt4o": "gpt-4o-2024-05-13",
    "gpt4": "gpt-4-turbo-2024-04-09",
    "gpt35": "gpt-3.5-turbo-0125"
}

GEMINI_MODELS: Dict[str, str] = {
    "pro": "gemini-pro",
    "ultra": "gemini-ultra",
    "pro-vision": "gemini-pro-vision"
}

DEEPSEEK_MODELS: Dict[str, str] = {
    "coder": "deepseek-coder", 
    "chat": "deepseek-chat"
}

# Default rate limits (requests per minute)
DEFAULT_RATE_LIMITS: Dict[str, float] = {
    "claude-3-7-opus-20250301": 2.0,
    "claude-3-7-sonnet-20250219": 3.0,
    "claude-3-5-haiku-20240307": 5.0,
    "gpt-4o-2024-05-13": 3.0,
    "gpt-4-turbo-2024-04-09": 2.0,
    "gpt-3.5-turbo-0125": 6.0,
    "gemini-pro": 6.0,
    "gemini-ultra": 2.0,
    "gemini-pro-vision": 6.0,
    "deepseek-coder": 5.0,
    "deepseek-chat": 5.0
}
```