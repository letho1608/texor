#!/usr/bin/env python3
"""
Test runner for the Nexor deep learning library.
Executes all test suites and provides a detailed report of test results.
"""

import unittest
import sys
import os
from typing import List, Tuple
import time

def collect_test_suites() -> List[unittest.TestSuite]:
    """Collect all test suites from the test directory"""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        loader.discover(start_dir, pattern='test_*.py')
    ]

def run_test_suite(suite: unittest.TestSuite) -> Tuple[int, int, float]:
    """Run a test suite and return the results"""
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return (
        result.testsRun,
        len(result.failures) + len(result.errors),
        end_time - start_time
    )

def print_separator(char: str = '-', length: int = 70):
    """Print a separator line"""
    print(char * length)

def main():
    print("\nNexor Test Runner")
    print_separator('=')
    print("Running all test suites...\n")
    
    total_tests = 0
    total_failures = 0
    total_time = 0
    
    # Collect and run all test suites
    suites = collect_test_suites()
    
    for suite in suites:
        tests_run, failures, duration = run_test_suite(suite)
        total_tests += tests_run
        total_failures += failures
        total_time += duration
        
        print_separator()
    
    # Print summary
    print("\nTest Summary")
    print_separator()
    print(f"Total tests run: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Success rate: {((total_tests - total_failures) / total_tests * 100):.2f}%")
    print(f"Total time: {total_time:.2f} seconds")
    print_separator()
    
    # Return appropriate exit code
    return 1 if total_failures > 0 else 0

if __name__ == '__main__':
    sys.exit(main())