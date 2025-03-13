#!/usr/bin/env python
"""
Response caching system for sentence extraction evaluations.
This module provides a simple caching mechanism to avoid redundant API calls.
"""

import hashlib
import json
import os
import time
from typing import Dict, Optional, Any


class ResponseCache:
    """
    Caches API responses to avoid redundant calls during evaluation.
    
    The cache uses SHA-256 hashing of prompt and input to create unique keys.
    Cached responses expire after a configurable time period (default 72 hours).
    """
    
    def __init__(self, cache_dir: str = "./cache", expiration_hours: int = 72):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory where cache files are stored
            expiration_hours: Number of hours before cached entries expire
        """
        self.cache_dir = cache_dir
        self.expiration_seconds = expiration_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, prompt: str, sentence: str, model: str) -> str:
        """
        Create a unique hash based on the prompt, sentence, and model.
        
        Args:
            prompt: The prompt template with the sentence inserted
            sentence: The sentence being processed
            model: The model identifier
            
        Returns:
            A hex string hash that uniquely identifies this request
        """
        # Include model in the key to handle different models separately
        key_material = f"{model}|{prompt}|{sentence}"
        return hashlib.sha256(key_material.encode()).hexdigest()
    
    def get(self, prompt: str, sentence: str, model: str) -> Optional[str]:
        """
        Retrieve a cached response if available and not expired.
        
        Args:
            prompt: The prompt template with the sentence inserted
            sentence: The sentence being processed
            model: The model identifier
            
        Returns:
            The cached response text if available, None otherwise
        """
        key = self._get_cache_key(prompt, sentence, model)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Only use cache if not expired
                    if time.time() - cached_data["timestamp"] < self.expiration_seconds:
                        return cached_data["response"]
                    else:
                        # Silently ignore expired cache entries
                        return None
            except (json.JSONDecodeError, KeyError, IOError):
                # If cache file is corrupted, ignore it
                return None
        return None
    
    def set(self, prompt: str, sentence: str, model: str, response: str) -> None:
        """
        Store a response in the cache.
        
        Args:
            prompt: The prompt template with the sentence inserted
            sentence: The sentence being processed
            model: The model identifier
            response: The model response to cache
        """
        key = self._get_cache_key(prompt, sentence, model)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "timestamp": time.time(),
                    "response": response,
                    "model": model
                }, f)
        except IOError:
            # Log warning if cache write fails, but continue execution
            print(f"Warning: Failed to write to cache file {cache_file}")
    
    def clear_expired(self) -> int:
        """
        Remove expired cache entries.
        
        Returns:
            Number of cache entries removed
        """
        cleared_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        cached_data = json.load(f)
                        if time.time() - cached_data["timestamp"] >= self.expiration_seconds:
                            os.remove(file_path)
                            cleared_count += 1
                except (json.JSONDecodeError, KeyError, IOError, OSError):
                    # Remove corrupt cache files
                    try:
                        os.remove(file_path)
                        cleared_count += 1
                    except OSError:
                        pass
        return cleared_count
    
    def clear_all(self) -> int:
        """
        Remove all cache entries.
        
        Returns:
            Number of cache entries removed
        """
        cleared_count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                try:
                    os.remove(os.path.join(self.cache_dir, filename))
                    cleared_count += 1
                except OSError:
                    pass
        return cleared_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dict with cache statistics (entry count, sizes, etc.)
        """
        entry_count = 0
        total_size = 0
        expired_count = 0
        models = set()
        
        current_time = time.time()
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                try:
                    size = os.path.getsize(file_path)
                    total_size += size
                    
                    with open(file_path, 'r') as f:
                        cached_data = json.load(f)
                        
                    # Count expired entries
                    if current_time - cached_data["timestamp"] >= self.expiration_seconds:
                        expired_count += 1
                    
                    # Track different models
                    if "model" in cached_data:
                        models.add(cached_data["model"])
                    
                    entry_count += 1
                except (OSError, json.JSONDecodeError, KeyError):
                    # Skip corrupt entries
                    pass
        
        return {
            "entry_count": entry_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "expired_count": expired_count,
            "active_count": entry_count - expired_count,
            "models": list(models)
        }


# Unit test function (can be run directly if this file is executed)
def test_cache():
    """Simple test of the cache functionality"""
    cache = ResponseCache(cache_dir="./test_cache")
    
    # Test basic caching
    test_prompt = "Extract information from: {SENTENCE}"
    test_sentence = "This is a test sentence."
    test_model = "test-model"
    test_response = "This is a test response."
    
    # Initially, should not be in cache
    assert cache.get(test_prompt, test_sentence, test_model) is None
    
    # Set the cache
    cache.set(test_prompt, test_sentence, test_model, test_response)
    
    # Now should be in cache
    assert cache.get(test_prompt, test_sentence, test_model) == test_response
    
    # Different input should not be in cache
    assert cache.get(test_prompt, "Different sentence.", test_model) is None
    assert cache.get(test_prompt, test_sentence, "different-model") is None
    
    # Get stats
    stats = cache.get_stats()
    assert stats["entry_count"] == 1
    
    # Clean up
    cache.clear_all()
    assert cache.get_stats()["entry_count"] == 0
    
    print("Cache tests passed!")


if __name__ == "__main__":
    test_cache()