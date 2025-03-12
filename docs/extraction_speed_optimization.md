# Extraction Speed Optimization Plan

This document outlines a comprehensive plan to optimize the extraction system's speed while maintaining accuracy. The plan is divided into phases for systematic implementation.

## Current Performance Assessment

Our sentence extraction system currently processes inputs sequentially, with several performance bottlenecks:

1. **Sequential API requests**: Each sentence is processed one at a time
2. **Fixed rate limiting**: Conservative delays between requests limit throughput
3. **Token-heavy prompts**: Verbose examples increase token count and processing time
4. **Redundant processing**: Same inputs may be repeatedly evaluated during testing
5. **Unoptimized comparison logic**: Deep comparison of results is computationally intensive

## Optimization Objectives

- **3-5x speed improvement** in overall extraction throughput
- **No reduction in extraction accuracy** for critical fields
- **Scalable architecture** to handle larger datasets
- **Resource-efficient processing** with lower computational cost

## Phase 1: Immediate Optimizations (1-2 Week Implementation)

### 1.1 Parallel API Request Processing

**Implementation**:
```python
from concurrent.futures import ThreadPoolExecutor

def process_batch_parallel(sentences, batch_size=10, max_workers=5):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sentence = {
            executor.submit(process_single_sentence, sentence): sentence 
            for sentence in sentences
        }
        for future in concurrent.futures.as_completed(future_to_sentence):
            results.append(future.result())
    return results
```

**Location**: Modify `evaluate_extraction()` in `claude_extraction_eval.py`

**Expected Impact**: 2-3x speedup for API-bound operations

### 1.2 Response Caching System

**Implementation**:
```python
import hashlib
import json
import os
import time

class ResponseCache:
    def __init__(self, cache_dir="./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, prompt, sentence):
        # Create a unique hash based on the prompt and input
        key_material = prompt + "|" + sentence
        return hashlib.sha256(key_material.encode()).hexdigest()
    
    def get(self, prompt, sentence):
        key = self._get_cache_key(prompt, sentence)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                # Only use cache if not expired (72 hours)
                if time.time() - cached_data["timestamp"] < 72 * 3600:
                    return cached_data["response"]
        return None
    
    def set(self, prompt, sentence, response):
        key = self._get_cache_key(prompt, sentence)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        with open(cache_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "response": response
            }, f)
```

**Location**: Create new module `extraction_cache.py` and integrate with `claude_extraction_eval.py`

**Expected Impact**: Near-instant results for previously processed inputs, 30-40% speedup during testing

### 1.3 Adaptive Rate Limiting

**Implementation**:
```python
import time
import threading

class AdaptiveRateLimiter:
    def __init__(self, requests_per_minute=60, burst=10):
        self.rate = requests_per_minute / 60.0  # convert to per second
        self.tokens = burst
        self.max_tokens = burst
        self.last_time = time.time()
        self.lock = threading.Lock()
        
    def wait(self):
        """Wait until a token is available for a request"""
        with self.lock:
            now = time.time()
            time_passed = now - self.last_time
            self.last_time = now
            
            # Add tokens based on time passed
            self.tokens += time_passed * self.rate
            self.tokens = min(self.tokens, self.max_tokens)
            
            if self.tokens < 1.0:
                # Calculate sleep time needed to get a token
                sleep_time = (1.0 - self.tokens) / self.rate
                time.sleep(sleep_time)
                self.tokens = 1.0
            
            # Consume a token
            self.tokens -= 1.0
```

**Location**: Create new module `rate_limiting.py` and integrate with API request code

**Expected Impact**: 20-30% improved throughput via optimized rate limiting

## Phase 2: Architectural Improvements (2-4 Week Implementation)

### 2.1 Tiered Prompt System

**Implementation**:
1. Create three versions of each prompt template:
   - **Fast**: Minimal examples, core instructions only (~40% token reduction)
   - **Standard**: Balanced examples and instructions (current approach)
   - **Comprehensive**: Complete examples for complex cases
   
2. Add speed_mode parameter to evaluation functions:
```python
def evaluate_extraction(sentences, prompt_template, model="claude-3-7-sonnet", 
                        speed_mode="standard", batch_size=10):
    """
    Evaluate extraction with selectable speed mode
    
    Args:
        speed_mode: "fast", "standard", or "comprehensive"
    """
    # Select appropriate prompt based on speed_mode
    if speed_mode == "fast":
        prompt_file = prompt_template.replace(".txt", "_fast.txt")
    elif speed_mode == "comprehensive":
        prompt_file = prompt_template.replace(".txt", "_comprehensive.txt")
    else:
        prompt_file = prompt_template
        
    # Continue with evaluation using selected prompt
```

**Location**: 
- Create new prompt variants in `prompt_enhanced/` directory
- Modify evaluation functions to support mode selection

**Expected Impact**: 25-35% faster processing in fast mode with minimal accuracy impact

### 2.2 Optimized JSON Parsing

**Implementation**:
```python
import re
import json

# Compile patterns once for better performance
JSON_PATTERN = re.compile(r'({[\s\S]*})')

def extract_json_efficiently(text):
    """Extract JSON from text using optimized pattern matching"""
    # First try direct pattern match (fastest)
    match = JSON_PATTERN.search(text)
    if match:
        try:
            json_str = match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Fall back to more comprehensive extraction if simple pattern fails
    try:
        return extract_json_fallback(text)
    except Exception:
        return None
```

**Location**: Replace JSON extraction in `claude_extraction_eval.py`

**Expected Impact**: 10-15% faster processing of model responses

### 2.3 Metrics Calculation Optimization

**Implementation**:
1. Refactor deep_compare() function to use more efficient algorithms
2. Implement early termination for obviously mismatched structures
3. Make detailed metrics calculation optional via configuration

```python
def deep_compare(expected, extracted, detailed_metrics=False):
    """
    Compare expected and extracted values with optimized algorithm
    
    Args:
        detailed_metrics: Whether to compute detailed metrics (slower)
    """
    # Quick comparison for trivial cases
    if expected == extracted:
        return 1.0, {}
        
    # For non-dict values, return simple match/no-match
    if not isinstance(expected, dict) or not isinstance(extracted, dict):
        return 1.0 if expected == extracted else 0.0, {}
    
    # Only perform expensive calculations if detailed metrics requested
    if not detailed_metrics:
        # Quick metrics for speed
        pass
    else:
        # Full detailed metrics (current implementation)
        pass
```

**Location**: Refactor comparison logic in `claude_extraction_eval.py`

**Expected Impact**: 15-20% faster overall processing, especially for large batches

## Phase 3: Advanced Scaling (4-8 Week Implementation)

### 3.1 Stream Processing for Large Datasets

**Implementation**:
1. Implement streaming evaluation for large datasets
2. Process and write results incrementally

```python
def stream_process_dataset(dataset_file, result_file, batch_size=50):
    """Process a large dataset in streaming fashion"""
    with open(dataset_file, 'r') as input_file, open(result_file, 'w') as output_file:
        # Write header for result file
        output_file.write('{"results": [\n')
        
        batch = []
        is_first = True
        
        # Process the dataset in streaming fashion
        for line in input_file:
            sentence = json.loads(line.strip())
            batch.append(sentence)
            
            if len(batch) >= batch_size:
                results = process_batch(batch)
                
                # Write results as they become available
                for result in results:
                    if not is_first:
                        output_file.write(',\n')
                    else:
                        is_first = False
                    json.dump(result, output_file)
                    output_file.flush()  # Ensure writing to disk
                
                batch = []
        
        # Process any remaining items
        if batch:
            results = process_batch(batch)
            for result in results:
                if not is_first:
                    output_file.write(',\n')
                else:
                    is_first = False
                json.dump(result, output_file)
        
        # Close the JSON array
        output_file.write('\n]}')
```

**Location**: Create new module `stream_processor.py` with streaming capabilities

**Expected Impact**: Ability to process datasets of any size with constant memory usage

### 3.2 Distributed Processing Support

**Implementation**:
1. Create worker system for distributed processing
2. Implement task queue with Redis or similar technology
3. Add checkpoint/resume capability

**Architecture Components**:
- **Task Manager**: Divides work into chunks and assigns to workers
- **Worker Nodes**: Process assigned chunks and return results
- **Result Aggregator**: Combines results from all workers

**Location**: Create new package `distributed/` with necessary modules

**Expected Impact**: Linear scaling with additional compute resources

## Phase 4: Production Hardening (Ongoing)

### 4.1 Performance Monitoring

**Implementation**:
1. Add detailed timing and performance metrics
2. Integrate with prompt tracker to record speed metrics

```python
class PerformanceTracker:
    def __init__(self):
        self.timings = {
            "api_calls": [],
            "json_parsing": [],
            "comparison": [],
            "total_processing": []
        }
        
    def record_timing(self, category, elapsed_seconds):
        self.timings[category].append(elapsed_seconds)
        
    def get_summary(self):
        summary = {}
        for category, times in self.timings.items():
            if times:
                summary[category] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times),
                    "count": len(times)
                }
        return summary
```

**Location**: Create new module `performance_monitoring.py` and integrate across the system

**Expected Impact**: Better understanding of performance bottlenecks and optimization opportunities

### 4.2 Configuration System

**Implementation**:
1. Create a unified configuration system for all optimization parameters
2. Allow presets for different usage scenarios (speed vs. accuracy)

```python
# Example configuration file
{
  "processing": {
    "mode": "parallel",  // sequential, parallel
    "max_workers": 8,
    "batch_size": 25
  },
  "rate_limiting": {
    "requests_per_minute": 300,
    "burst_capacity": 50,
    "adaptive": true
  },
  "caching": {
    "enabled": true,
    "location": "./cache",
    "expiration_hours": 72
  },
  "prompts": {
    "speed_mode": "fast",  // fast, standard, comprehensive
    "use_examples": true
  },
  "metrics": {
    "detailed_comparison": false,
    "performance_tracking": true
  }
}
```

**Location**: Create configuration management system

**Expected Impact**: Flexible optimization based on specific use cases

## Implementation Timeline

| Phase | Feature | Timeline | Priority | Complexity | Impact |
|-------|---------|----------|----------|------------|--------|
| 1 | Parallel API Processing | 1 week | HIGH | Medium | ★★★★☆ |
| 1 | Response Caching | 1 week | HIGH | Low | ★★★★☆ |
| 1 | Adaptive Rate Limiting | 1 week | MEDIUM | Low | ★★★☆☆ |
| 2 | Tiered Prompt System | 2 weeks | HIGH | Medium | ★★★★☆ |
| 2 | Optimized JSON Parsing | 1 week | MEDIUM | Low | ★★☆☆☆ |
| 2 | Metrics Calculation Optimization | 2 weeks | MEDIUM | High | ★★★☆☆ |
| 3 | Stream Processing | 3 weeks | LOW | High | ★★☆☆☆ |
| 3 | Distributed Processing | 4 weeks | LOW | Very High | ★★★★★ |
| 4 | Performance Monitoring | 2 weeks | MEDIUM | Medium | ★★☆☆☆ |
| 4 | Configuration System | 2 weeks | LOW | Medium | ★★☆☆☆ |

## Expected Outcomes

After implementing all phases, we expect:

1. **Processing Speed**: 3-5x faster extraction processing
2. **Resource Efficiency**: Lower computational and API cost per evaluation
3. **Scalability**: Ability to process datasets of any size
4. **Flexibility**: Configurable tradeoffs between speed and comprehensive analysis

## Success Metrics

We will measure success based on:

1. **Throughput**: Sentences processed per minute
2. **Latency**: Average time to process a single sentence
3. **Resource Usage**: API tokens consumed per evaluation
4. **Accuracy Retention**: Ensuring optimizations don't reduce extraction accuracy

## Conclusion

This optimization plan provides a systematic approach to significantly improving the speed of our extraction system while maintaining accuracy. The phased implementation allows for incremental improvements with each phase delivering tangible benefits.

By prioritizing the high-impact, low-complexity optimizations in Phase 1, we can achieve substantial improvements quickly, while the later phases provide a path to enterprise-scale processing capabilities.