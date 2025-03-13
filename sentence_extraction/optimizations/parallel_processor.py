#!/usr/bin/env python
"""
Parallel processing for extraction evaluation.
This module implements concurrent processing of sentences for extraction tasks.
"""

import time
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm


class ParallelProcessor:
    """
    Manages parallel processing for extraction tasks.
    
    Uses ThreadPoolExecutor to process multiple sentences concurrently, 
    with progress tracking and error handling.
    """
    
    def __init__(self, max_workers: int = 5, show_progress: bool = True):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads
            show_progress: Whether to show a progress bar
        """
        self.max_workers = max_workers
        self.show_progress = show_progress
        self.stats = {
            "processed_items": 0,
            "successful_items": 0,
            "failed_items": 0,
            "total_processing_time": 0,
            "start_time": None,
            "end_time": None
        }
    
    def process_batch(self, 
                     items: List[Any], 
                     process_func: Callable[[Any], Any],
                     desc: str = "Processing batch") -> List[Any]:
        """
        Process a batch of items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            desc: Description for the progress bar
            
        Returns:
            List of results from processing each item
        """
        results = []
        self.stats["start_time"] = time.time()
        
        # Create progress bar if enabled
        progress_bar = None
        if self.show_progress:
            progress_bar = tqdm(total=len(items), desc=desc)
        
        # Process items in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._safe_process, process_func, item): item 
                for item in items
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_item):
                result = future.result()
                results.append(result)
                
                # Update progress bar if enabled
                if progress_bar:
                    progress_bar.update(1)
                
                # Update stats
                self.stats["processed_items"] += 1
                if result.get("error") is None:
                    self.stats["successful_items"] += 1
                else:
                    self.stats["failed_items"] += 1
        
        # Close progress bar if enabled
        if progress_bar:
            progress_bar.close()
        
        self.stats["end_time"] = time.time()
        self.stats["total_processing_time"] = self.stats["end_time"] - self.stats["start_time"]
        
        return results
    
    def _safe_process(self, process_func: Callable[[Any], Any], item: Any) -> Dict[str, Any]:
        """
        Safely process an item, catching and recording any exceptions.
        
        Args:
            process_func: Function to process the item
            item: Item to process
            
        Returns:
            Dictionary with processed result or error information
        """
        start_time = time.time()
        try:
            result = process_func(item)
            processing_time = time.time() - start_time
            
            # If result is already a dict, add processing_time
            if isinstance(result, dict):
                result["processing_time"] = processing_time
                return result
            else:
                # Wrap non-dict results
                return {
                    "result": result,
                    "processing_time": processing_time
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Create an error result
            error_result = {
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time": processing_time
            }
            
            # Include the original item for context
            if isinstance(item, dict):
                # Only include a subset of fields if item is large
                if "sentence" in item:
                    error_result["sentence"] = item["sentence"]
                    
            return error_result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dict with processing statistics
        """
        stats = self.stats.copy()
        
        # Add derived statistics
        if stats["processed_items"] > 0:
            stats["success_rate"] = stats["successful_items"] / stats["processed_items"]
            stats["avg_processing_time"] = stats["total_processing_time"] / stats["processed_items"]
            
            if stats["end_time"] and stats["start_time"]:
                elapsed = stats["end_time"] - stats["start_time"]
                stats["throughput"] = stats["processed_items"] / elapsed if elapsed > 0 else 0
                stats["throughput_per_minute"] = stats["throughput"] * 60
        
        return stats


def process_data_parallel(data_items: List[Dict[str, Any]], 
                         process_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                         max_workers: int = 5,
                         batch_size: Optional[int] = None,
                         show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience function to process data items in parallel.
    
    Args:
        data_items: List of data items to process
        process_func: Function to process each item
        max_workers: Maximum number of worker threads
        batch_size: Number of items per batch (None for single batch)
        show_progress: Whether to show progress bar
    
    Returns:
        List of processed results
    """
    processor = ParallelProcessor(max_workers=max_workers, show_progress=show_progress)
    
    # Process in a single batch if batch_size is None
    if batch_size is None:
        return processor.process_batch(data_items, process_func)
    
    # Process in multiple batches
    results = []
    for i in range(0, len(data_items), batch_size):
        batch = data_items[i:i+batch_size]
        batch_results = processor.process_batch(
            batch, 
            process_func, 
            desc=f"Processing batch {i//batch_size + 1}/{(len(data_items)-1)//batch_size + 1}"
        )
        results.extend(batch_results)
    
    return results


# Test function
def test_parallel_processor():
    """Simple test of the parallel processor functionality"""
    
    def dummy_process(item):
        """Dummy processing function that sleeps and returns a result"""
        # Simulate processing time
        delay = item.get("delay", 0.1)
        time.sleep(delay)
        
        # Simulate occasional errors
        if item.get("should_fail", False):
            raise ValueError("Simulated error")
        
        return {
            "input": item,
            "output": f"Processed: {item.get('id', 'unknown')}"
        }
    
    # Create test items
    test_items = [
        {"id": i, "delay": 0.1 + (i % 3) * 0.1, "should_fail": i % 5 == 0}
        for i in range(20)
    ]
    
    print("Testing ParallelProcessor...")
    
    # Process items with different worker counts
    for workers in [1, 3, 5]:
        print(f"\nTesting with {workers} workers:")
        processor = ParallelProcessor(max_workers=workers)
        
        start = time.time()
        results = processor.process_batch(test_items, dummy_process, 
                                          desc=f"Testing {workers} workers")
        duration = time.time() - start
        
        # Count successes and failures
        successes = sum(1 for r in results if "error" not in r)
        failures = sum(1 for r in results if "error" in r)
        
        print(f"Processed {len(results)} items in {duration:.2f} seconds")
        print(f"Successes: {successes}, Failures: {failures}")
        print(f"Stats: {processor.get_stats()}")
    
    # Test the convenience function
    print("\nTesting process_data_parallel function with batching:")
    start = time.time()
    results = process_data_parallel(
        test_items, 
        dummy_process,
        max_workers=3,
        batch_size=5
    )
    duration = time.time() - start
    
    print(f"Processed {len(results)} items in {duration:.2f} seconds")
    print("Parallel processor tests completed!")


if __name__ == "__main__":
    test_parallel_processor()