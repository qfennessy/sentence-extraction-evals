#!/usr/bin/env python
"""
Adaptive rate limiting for API requests.
This module provides a token bucket algorithm implementation for rate limiting.
"""

import time
import threading
from typing import Dict, Any, Optional


class AdaptiveRateLimiter:
    """
    Implements a token bucket algorithm for adaptive rate limiting.
    
    This allows bursts of requests while maintaining a sustainable average rate.
    Thread-safe for concurrent usage in multi-threaded environments.
    """
    
    def __init__(self, requests_per_minute: float = 60.0, burst_capacity: int = 10):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Sustainable rate of requests (per minute)
            burst_capacity: Maximum number of tokens that can be accumulated
        """
        self.rate = requests_per_minute / 60.0  # Convert to per second
        self.tokens = burst_capacity  # Start with full bucket
        self.max_tokens = burst_capacity
        self.last_time = time.time()
        self.lock = threading.Lock()
        self.stats = {
            "total_requests": 0,
            "wait_events": 0,
            "total_wait_time": 0,
            "max_wait_time": 0,
            "last_minute_requests": 0,
            "last_minute_timestamp": time.time()
        }
    
    def wait(self) -> float:
        """
        Wait until a token is available for a request.
        
        Returns:
            Float indicating the number of seconds waited (0 if no wait)
        """
        wait_time = 0
        
        with self.lock:
            now = time.time()
            time_passed = now - self.last_time
            self.last_time = now
            
            # Add tokens based on time passed
            self.tokens += time_passed * self.rate
            self.tokens = min(self.tokens, self.max_tokens)
            
            # Update stats
            self.stats["total_requests"] += 1
            
            # Reset last minute counter if needed
            if now - self.stats["last_minute_timestamp"] >= 60:
                self.stats["last_minute_requests"] = 0
                self.stats["last_minute_timestamp"] = now
            self.stats["last_minute_requests"] += 1
            
            if self.tokens < 1.0:
                # Calculate sleep time needed to get a token
                wait_time = (1.0 - self.tokens) / self.rate
                
                # Update wait stats before sleeping
                self.stats["wait_events"] += 1
                self.stats["total_wait_time"] += wait_time
                self.stats["max_wait_time"] = max(self.stats["max_wait_time"], wait_time)
                
                # Release lock during sleep
                self.lock.release()
                try:
                    time.sleep(wait_time)
                finally:
                    # Re-acquire lock after sleep
                    self.lock.acquire()
                
                # Set tokens to 1.0 (we're consuming one token after the wait)
                self.tokens = 1.0
            
            # Consume a token
            self.tokens -= 1.0
        
        return wait_time
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the rate limiter usage.
        
        Returns:
            Dict with usage statistics
        """
        with self.lock:
            # Calculate derived stats
            avg_wait_time = 0
            if self.stats["wait_events"] > 0:
                avg_wait_time = self.stats["total_wait_time"] / self.stats["wait_events"]
            
            # Copy stats to avoid modification during access
            stats_copy = self.stats.copy()
            
            # Add current tokens and derived stats
            stats_copy.update({
                "current_tokens": self.tokens,
                "target_rate_per_minute": self.rate * 60,
                "burst_capacity": self.max_tokens,
                "avg_wait_time": avg_wait_time,
                "wait_percentage": (self.stats["wait_events"] / max(1, self.stats["total_requests"])) * 100
            })
            
            return stats_copy
    
    def update_rate(self, new_requests_per_minute: float) -> None:
        """
        Update the rate limit dynamically.
        
        Args:
            new_requests_per_minute: New sustainable rate (per minute)
        """
        with self.lock:
            self.rate = new_requests_per_minute / 60.0


class ModelRateLimiter:
    """
    Manages separate rate limiters for different models.
    """
    
    def __init__(self, default_rates: Optional[Dict[str, float]] = None):
        """
        Initialize with default rates for known models.
        
        Args:
            default_rates: Dictionary mapping model IDs to requests per minute
        """
        self.limiters = {}
        
        # Default rate limits for different models (requests per minute)
        # These are conservative defaults - adjust based on API quotas
        self.default_rates = {
            # Claude models
            "claude-3-opus-20240229": 60,     # 1 request per second
            "claude-3-7-sonnet-20250219": 120,  # 2 requests per second
            "claude-3-haiku-20240307": 240,    # 4 requests per second
            
            # OpenAI models
            "gpt-4o": 60,                      # 1 request per second
            "gpt-4-turbo-2024-04-09": 60,      # 1 request per second 
            "gpt-3.5-turbo-0125": 180,         # 3 requests per second
            
            # Gemini models
            "gemini-pro": 120,                 # 2 requests per second
            "gemini-pro-vision": 60,           # 1 request per second
            "gemini-ultra": 60,                # 1 request per second
            
            # DeepSeek models
            "deepseek-coder": 60,              # 1 request per second
            "deepseek-chat": 60                # 1 request per second
        }
        
        # Update with any provided overrides
        if default_rates:
            self.default_rates.update(default_rates)
    
    def get_limiter(self, model: str) -> AdaptiveRateLimiter:
        """
        Get the appropriate rate limiter for a model.
        
        Args:
            model: Model identifier
            
        Returns:
            Rate limiter instance for the specified model
        """
        if model not in self.limiters:
            # Use specific model rate if available, otherwise use a conservative default
            rate = self.default_rates.get(model, 60)  # Default to 60 rpm if unknown
            burst = min(20, rate // 3)  # Reasonable burst capacity based on rate
            self.limiters[model] = AdaptiveRateLimiter(requests_per_minute=rate, burst_capacity=burst)
        
        return self.limiters[model]
    
    def wait(self, model: str) -> float:
        """
        Wait for the rate limit for the specified model.
        
        Args:
            model: Model identifier
            
        Returns:
            Time waited in seconds
        """
        return self.get_limiter(model).wait()
    
    def get_stats(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for one or all rate limiters.
        
        Args:
            model: Optional model ID. If provided, returns stats for that model only.
                  If None, returns stats for all models.
                  
        Returns:
            Dict with usage statistics for the requested model(s)
        """
        if model:
            if model in self.limiters:
                return self.limiters[model].get_stats()
            return {"error": f"No rate limiter found for model: {model}"}
        
        # Return stats for all models
        return {model: limiter.get_stats() for model, limiter in self.limiters.items()}


# Test function
def test_rate_limiter():
    """Simple test of the rate limiter functionality"""
    # Test basic rate limiting
    limiter = AdaptiveRateLimiter(requests_per_minute=60, burst_capacity=10)
    
    print("Testing AdaptiveRateLimiter...")
    # Make a burst of requests
    start = time.time()
    for i in range(15):
        wait_time = limiter.wait()
        print(f"Request {i+1}: waited {wait_time:.4f}s, tokens remaining: {limiter.tokens:.2f}")
    
    duration = time.time() - start
    print(f"Burst test completed in {duration:.2f} seconds")
    print(f"Stats: {limiter.get_stats()}")
    
    # Test ModelRateLimiter
    print("\nTesting ModelRateLimiter...")
    model_limiter = ModelRateLimiter()
    
    # Test with different models
    models = ["claude-3-7-sonnet-20250219", "gpt-4o", "unknown-model"]
    
    for model in models:
        print(f"\nTesting rate limiting for {model}")
        wait_time = model_limiter.wait(model)
        print(f"Wait time: {wait_time:.4f}s")
        print(f"Stats: {model_limiter.get_stats(model)}")
    
    print("\nAll models:")
    all_stats = model_limiter.get_stats()
    print(f"Number of active limiters: {len(all_stats)}")
    
    print("Rate limiter tests completed!")


if __name__ == "__main__":
    test_rate_limiter()