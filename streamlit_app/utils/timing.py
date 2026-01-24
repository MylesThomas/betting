"""
Timing utilities for performance monitoring.

Provides decorators to time function execution and log results.
"""

import time
import functools
from datetime import datetime


def timed(func):
    """
    Decorator to time function execution and print results.
    
    Usage:
        @timed
        def my_function():
            # do work
            pass
    
    Output:
        ⏱️  my_function took 2.34 seconds
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        print(f"\n⏱️  [{timestamp}] Starting {func.__name__}...")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed = end_time - start_time
        timestamp_end = datetime.now().strftime('%H:%M:%S')
        
        print(f"✅ [{timestamp_end}] {func.__name__} completed in {elapsed:.2f} seconds")
        
        return result
    
    return wrapper


def timed_section(name):
    """
    Context manager to time a code section.
    
    Usage:
        with timed_section("Loading data"):
            # do work
            pass
    
    Output:
        ⏱️  [14:30:15] Starting: Loading data
        ✅ [14:30:17] Completed: Loading data (2.34 seconds)
    """
    class TimedSection:
        def __init__(self, section_name):
            self.section_name = section_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"\n⏱️  [{timestamp}] Starting: {self.section_name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = time.time()
            elapsed = end_time - self.start_time
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"✅ [{timestamp}] Completed: {self.section_name} ({elapsed:.2f} seconds)")
    
    return TimedSection(name)
