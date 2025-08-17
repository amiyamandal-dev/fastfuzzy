#!/usr/bin/env python3
"""
Test to see if we can run the benchmark script structure without calling functions
"""

import time
import random
import string
import statistics
import json
import argparse
from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import sys
import traceback

# Import the libraries to benchmark
try:
    import fastfuzzy
    fastfuzzy_AVAILABLE = True
except ImportError:
    fastfuzzy_AVAILABLE = False
    print("WARNING: fastfuzzy not available")

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    library: str
    operation: str
    times: List[float]
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    operations_per_second: float
    relative_speed: float = 1.0

def generate_test_strings(size: int, min_len: int = 5, max_len: int = 50) -> List[str]:
    """Generate random test strings."""
    strings = []
    for _ in range(size):
        length = random.randint(min_len, max_len)
        s = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
        strings.append(s.strip())
    return strings

def create_similar_strings(base_strings: List[str], similarity: float = 0.8) -> List[str]:
    """Create strings similar to base strings by introducing random changes."""
    similar = []
    for s in base_strings:
        if not s:
            similar.append(s)
            continue

        # Determine number of changes to make
        changes = max(1, int(len(s) * (1 - similarity)))
        chars = list(s)

        for _ in range(changes):
            if not chars:
                break

            change_type = random.choice(['substitute', 'insert', 'delete'])

            if change_type == 'substitute' and chars:
                idx = random.randrange(len(chars))
                chars[idx] = random.choice(string.ascii_letters)
            elif change_type == 'insert':
                idx = random.randrange(len(chars) + 1)
                chars.insert(idx, random.choice(string.ascii_letters))
            elif change_type == 'delete' and chars:
                idx = random.randrange(len(chars))
                chars.pop(idx)

        similar.append(''.join(chars))

    return similar

class BenchmarkSuite:
    """Main benchmark suite class."""

    def __init__(self, size: int = 1000, iterations: int = 5):
        self.size = size
        self.iterations = iterations
        self.results: List[BenchmarkResult] = []

        # Generate test data
        print(f"Generating {size} test strings...")
        self.test_strings = generate_test_strings(size)
        self.similar_strings = create_similar_strings(self.test_strings, 0.8)

        # Create query and choice sets
        self.queries = self.test_strings[:size//10]  # 10% as queries
        self.choices = self.test_strings[size//10:]  # 90% as choices

        print(f"Generated {len(self.queries)} queries and {len(self.choices)} choices")

    def run_single_comparison_benchmarks(self):
        """Benchmark single string comparisons."""
        print("\n=== Single String Comparison Benchmarks ===")

        # Select test pairs
        test_pairs = list(zip(self.test_strings[:5], self.similar_strings[:5]))

        # Just print what we would test, don't actually call the functions
        benchmarks = []

        # fastfuzzy benchmarks
        if fastfuzzy_AVAILABLE:
            benchmarks.extend([
                ("fastfuzzy", "ratio"),
                ("fastfuzzy", "partial_ratio"),
                ("fastfuzzy", "token_sort_ratio"),
                ("fastfuzzy", "levenshtein"),
                ("fastfuzzy", "jaro_winkler"),
            ])

        # Run benchmarks
        for library, operation in benchmarks:
            print(f"  {library} - {operation}... would test with {len(test_pairs)} pairs")
            # Skip actual function calls to avoid segfault
            print(f"  Would run: {[f'{library}.{operation}(s1, s2, 0.5)' for s1, s2 in test_pairs[:2]]}")

    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        print("fastfuzzy Performance Benchmark Suite")
        print("="*50)
        print(f"Dataset size: {self.size}")
        print(f"Iterations per test: {self.iterations}")
        print(f"Libraries available: fastfuzzy={fastfuzzy_AVAILABLE}")

        try:
            self.run_single_comparison_benchmarks()
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        except Exception as e:
            print(f"\nBenchmark failed with error: {e}")
            traceback.print_exc()

def main():
    """Main function."""
    # Run benchmarks
    suite = BenchmarkSuite(size=100, iterations=1)
    suite.run_all_benchmarks()

if __name__ == "__main__":
    main()