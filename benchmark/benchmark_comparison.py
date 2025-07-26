#!/usr/bin/env python3
"""
fast_fuzzy Performance Benchmark Suite

This script compares fast_fuzzy performance against other popular fuzzy string
matching libraries including RapidFuzz, FuzzyWuzzy, and python-Levenshtein.

Usage:
    python benchmark_comparison.py [--size SIZE] [--iterations ITER] [--output OUTPUT]

Example:
    python benchmark_comparison.py --size 10000 --iterations 5 --output benchmark_results.json
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
    import fast_fuzzy
    fast_fuzzy_AVAILABLE = True
except ImportError:
    fast_fuzzy_AVAILABLE = False
    print("WARNING: fast_fuzzy not available")

try:
    import rapidfuzz
    from rapidfuzz import fuzz as rf_fuzz, process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("WARNING: RapidFuzz not available")

try:
    from fuzzywuzzy import fuzz as fw_fuzz, process as fw_process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("WARNING: FuzzyWuzzy not available")

try:
    import Levenshtein
    PYTHON_LEVENSHTEIN_AVAILABLE = True
except ImportError:
    PYTHON_LEVENSHTEIN_AVAILABLE = False
    print("WARNING: python-Levenshtein not available")

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

def time_function(func: Callable, *args, iterations: int = 5, **kwargs) -> List[float]:
    """Time a function over multiple iterations."""
    times = []

    # Warmup
    try:
        func(*args, **kwargs)
    except Exception:
        pass

    for _ in range(iterations):
        start_time = time.perf_counter()
        try:
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except Exception as e:
            print(f"Error in function {func.__name__}: {e}")
            times.append(float('inf'))

    return times

def calculate_stats(times: List[float], num_operations: int = 1) -> Dict[str, float]:
    """Calculate timing statistics."""
    valid_times = [t for t in times if t != float('inf')]

    if not valid_times:
        return {
            'mean': float('inf'),
            'std': 0.0,
            'min': float('inf'),
            'max': float('inf'),
            'ops_per_sec': 0.0
        }

    mean_time = statistics.mean(valid_times)
    std_time = statistics.stdev(valid_times) if len(valid_times) > 1 else 0.0
    min_time = min(valid_times)
    max_time = max(valid_times)
    ops_per_sec = num_operations / mean_time if mean_time > 0 else 0.0

    return {
        'mean': mean_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'ops_per_sec': ops_per_sec
    }

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
        test_pairs = list(zip(self.test_strings[:100], self.similar_strings[:100]))

        benchmarks = []

        # fast_fuzzy benchmarks
        if fast_fuzzy_AVAILABLE:
            benchmarks.extend([
                ("fast_fuzzy", "ratio", lambda: [fast_fuzzy.ratio(s1, s2, 0.5) for s1, s2 in test_pairs]),
                ("fast_fuzzy", "partial_ratio", lambda: [fast_fuzzy.partial_ratio(s1, s2, 0.5) for s1, s2 in test_pairs]),
                ("fast_fuzzy", "token_sort_ratio", lambda: [fast_fuzzy.token_sort_ratio(s1, s2, 0.5) for s1, s2 in test_pairs]),
                ("fast_fuzzy", "levenshtein", lambda: [fast_fuzzy.levenshtein(s1, s2) for s1, s2 in test_pairs]),
                ("fast_fuzzy", "jaro_winkler", lambda: [fast_fuzzy.jaro_winkler(s1, s2) for s1, s2 in test_pairs]),
            ])

        # RapidFuzz benchmarks
        if RAPIDFUZZ_AVAILABLE:
            benchmarks.extend([
                ("RapidFuzz", "ratio", lambda: [rf_fuzz.ratio(s1, s2) for s1, s2 in test_pairs]),
                ("RapidFuzz", "partial_ratio", lambda: [rf_fuzz.partial_ratio(s1, s2) for s1, s2 in test_pairs]),
                ("RapidFuzz", "token_sort_ratio", lambda: [rf_fuzz.token_sort_ratio(s1, s2) for s1, s2 in test_pairs]),
                ("RapidFuzz", "levenshtein", lambda: [rapidfuzz.distance.Levenshtein.distance(s1, s2) for s1, s2 in test_pairs]),
                ("RapidFuzz", "jaro_winkler", lambda: [rapidfuzz.distance.JaroWinkler.similarity(s1, s2) for s1, s2 in test_pairs]),
            ])

        # FuzzyWuzzy benchmarks
        if FUZZYWUZZY_AVAILABLE:
            benchmarks.extend([
                ("FuzzyWuzzy", "ratio", lambda: [fw_fuzz.ratio(s1, s2) for s1, s2 in test_pairs]),
                ("FuzzyWuzzy", "partial_ratio", lambda: [fw_fuzz.partial_ratio(s1, s2) for s1, s2 in test_pairs]),
                ("FuzzyWuzzy", "token_sort_ratio", lambda: [fw_fuzz.token_sort_ratio(s1, s2) for s1, s2 in test_pairs]),
            ])

        # python-Levenshtein benchmarks
        if PYTHON_LEVENSHTEIN_AVAILABLE:
            benchmarks.extend([
                ("python-Levenshtein", "distance", lambda: [Levenshtein.distance(s1, s2) for s1, s2 in test_pairs]),
                ("python-Levenshtein", "ratio", lambda: [Levenshtein.ratio(s1, s2) for s1, s2 in test_pairs]),
                ("python-Levenshtein", "jaro_winkler", lambda: [Levenshtein.jaro_winkler(s1, s2) for s1, s2 in test_pairs]),
            ])

        # Run benchmarks
        for library, operation, func in benchmarks:
            print(f"  {library} - {operation}...", end=" ", flush=True)
            times = time_function(func, iterations=self.iterations)
            stats = calculate_stats(times, len(test_pairs))

            result = BenchmarkResult(
                name=f"{library}_{operation}",
                library=library,
                operation=operation,
                times=times,
                mean_time=stats['mean'],
                std_time=stats['std'],
                min_time=stats['min'],
                max_time=stats['max'],
                operations_per_second=stats['ops_per_sec']
            )

            self.results.append(result)
            print(f"{stats['ops_per_sec']:.0f} ops/sec")

    def run_batch_processing_benchmarks(self):
        """Benchmark batch processing operations."""
        print("\n=== Batch Processing Benchmarks ===")

        # Use smaller datasets for batch operations
        batch_queries = self.queries[:10]
        batch_choices = self.choices[:1000]

        benchmarks = []

        # fast_fuzzy batch benchmarks
        if fast_fuzzy_AVAILABLE:
            benchmarks.extend([
                ("fast_fuzzy", "batch_ratio", lambda: fast_fuzzy.batch_ratio(batch_queries, batch_choices)),
                ("fast_fuzzy", "batch_levenshtein", lambda: fast_fuzzy.batch_levenshtein(batch_queries, batch_choices)),
            ])

        # RapidFuzz batch benchmarks
        if RAPIDFUZZ_AVAILABLE:
            benchmarks.extend([
                ("RapidFuzz", "cdist_ratio", lambda: rf_process.cdist(batch_queries, batch_choices, scorer=rf_fuzz.ratio)),
                ("RapidFuzz", "cdist_levenshtein", lambda: rf_process.cdist(batch_queries, batch_choices, scorer=rapidfuzz.distance.Levenshtein.normalized_similarity)),
            ])

        # Run benchmarks
        for library, operation, func in benchmarks:
            print(f"  {library} - {operation}...", end=" ", flush=True)
            times = time_function(func, iterations=max(1, self.iterations // 3))  # Fewer iterations for batch
            num_ops = len(batch_queries) * len(batch_choices)
            stats = calculate_stats(times, num_ops)

            result = BenchmarkResult(
                name=f"{library}_{operation}",
                library=library,
                operation=operation,
                times=times,
                mean_time=stats['mean'],
                std_time=stats['std'],
                min_time=stats['min'],
                max_time=stats['max'],
                operations_per_second=stats['ops_per_sec']
            )

            self.results.append(result)
            print(f"{stats['ops_per_sec']:.0f} ops/sec")

    def run_fuzzy_search_benchmarks(self):
        """Benchmark fuzzy search operations."""
        print("\n=== Fuzzy Search Benchmarks ===")

        # Use subset for search benchmarks
        search_queries = self.queries[:20]
        search_choices = self.choices[:5000]

        benchmarks = []

        # fast_fuzzy search benchmarks
        if fast_fuzzy_AVAILABLE:
            benchmarks.extend([
                ("fast_fuzzy", "extract", lambda: [fast_fuzzy.process.extract(q, search_choices, limit=5) for q in search_queries]),
                ("fast_fuzzy", "extractOne", lambda: [fast_fuzzy.process.extractOne(q, search_choices) for q in search_queries]),
            ])

        # RapidFuzz search benchmarks
        if RAPIDFUZZ_AVAILABLE:
            benchmarks.extend([
                ("RapidFuzz", "extract", lambda: [rf_process.extract(q, search_choices, limit=5) for q in search_queries]),
                ("RapidFuzz", "extractOne", lambda: [rf_process.extractOne(q, search_choices) for q in search_queries]),
            ])

        # FuzzyWuzzy search benchmarks
        if FUZZYWUZZY_AVAILABLE:
            benchmarks.extend([
                ("FuzzyWuzzy", "extract", lambda: [fw_process.extract(q, search_choices, limit=5) for q in search_queries]),
                ("FuzzyWuzzy", "extractOne", lambda: [fw_process.extractOne(q, search_choices) for q in search_queries]),
            ])

        # Run benchmarks
        for library, operation, func in benchmarks:
            print(f"  {library} - {operation}...", end=" ", flush=True)
            times = time_function(func, iterations=max(1, self.iterations // 2))
            num_ops = len(search_queries) * len(search_choices)
            stats = calculate_stats(times, num_ops)

            result = BenchmarkResult(
                name=f"{library}_{operation}",
                library=library,
                operation=operation,
                times=times,
                mean_time=stats['mean'],
                std_time=stats['std'],
                min_time=stats['min'],
                max_time=stats['max'],
                operations_per_second=stats['ops_per_sec']
            )

            self.results.append(result)
            print(f"{stats['ops_per_sec']:.0f} ops/sec")

    def run_memory_efficiency_test(self):
        """Test memory efficiency with large datasets."""
        print("\n=== Memory Efficiency Test ===")

        try:
            import psutil
            import os

            def get_memory_usage():
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / 1024 / 1024  # MB

            # Large dataset
            large_queries = self.queries[:5]
            large_choices = self.test_strings  # Full dataset

            libraries_to_test = []

            if fast_fuzzy_AVAILABLE:
                libraries_to_test.append(("fast_fuzzy", lambda q: fast_fuzzy.process.extract(q, large_choices, limit=10)))

            if RAPIDFUZZ_AVAILABLE:
                libraries_to_test.append(("RapidFuzz", lambda q: rf_process.extract(q, large_choices, limit=10)))

            for library_name, extract_func in libraries_to_test:
                print(f"  {library_name} memory test...", end=" ", flush=True)

                initial_memory = get_memory_usage()

                start_time = time.perf_counter()
                for query in large_queries:
                    extract_func(query)
                end_time = time.perf_counter()

                final_memory = get_memory_usage()
                memory_used = final_memory - initial_memory

                print(f"Memory: {memory_used:.1f}MB, Time: {end_time - start_time:.2f}s")

        except ImportError:
            print("  psutil not available, skipping memory test")

    def calculate_relative_speeds(self):
        """Calculate relative speeds compared to baseline."""
        # Group results by operation
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)

        # Calculate relative speeds for each operation
        for operation, results in operations.items():
            if not results:
                continue

            # Find fastest result as baseline
            fastest = min(results, key=lambda r: r.mean_time)
            baseline_speed = fastest.operations_per_second

            # Calculate relative speeds
            for result in results:
                if baseline_speed > 0 and result.operations_per_second > 0:
                    result.relative_speed = result.operations_per_second / baseline_speed
                else:
                    result.relative_speed = 0.0

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        # Group by operation
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)

        for operation, results in operations.items():
            print(f"\n{operation.upper()}")
            print("-" * 40)

            # Sort by operations per second (descending)
            results.sort(key=lambda r: r.operations_per_second, reverse=True)

            for i, result in enumerate(results):
                speed_indicator = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
                print(f"{speed_indicator} {result.library:15} {result.operations_per_second:>12,.0f} ops/sec "
                      f"({result.relative_speed:>5.2f}x) ± {result.std_time*1000:>6.1f}ms")

        # Overall performance summary
        print(f"\n{'OVERALL PERFORMANCE':^80}")
        print("-" * 80)

        library_totals = {}
        for result in self.results:
            if result.library not in library_totals:
                library_totals[result.library] = []
            library_totals[result.library].append(result.operations_per_second)

        library_averages = {
            lib: statistics.mean(speeds)
            for lib, speeds in library_totals.items()
        }

        sorted_libraries = sorted(library_averages.items(), key=lambda x: x[1], reverse=True)

        for i, (library, avg_speed) in enumerate(sorted_libraries):
            speed_indicator = "🏆" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(f"{speed_indicator} {library:15} {avg_speed:>12,.0f} ops/sec average")

    def save_results(self, filename: str):
        """Save results to JSON file."""
        data = {
            'benchmark_info': {
                'size': self.size,
                'iterations': self.iterations,
                'timestamp': time.time(),
            },
            'results': []
        }

        for result in self.results:
            data['results'].append({
                'name': result.name,
                'library': result.library,
                'operation': result.operation,
                'mean_time': result.mean_time,
                'std_time': result.std_time,
                'min_time': result.min_time,
                'max_time': result.max_time,
                'operations_per_second': result.operations_per_second,
                'relative_speed': result.relative_speed,
                'times': result.times,
            })

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {filename}")

    def run_all_benchmarks(self):
        """Run all benchmark suites."""
        print("fast_fuzzy Performance Benchmark Suite")
        print("="*50)
        print(f"Dataset size: {self.size}")
        print(f"Iterations per test: {self.iterations}")
        print(f"Libraries available: fast_fuzzy={fast_fuzzy_AVAILABLE}, "
              f"RapidFuzz={RAPIDFUZZ_AVAILABLE}, FuzzyWuzzy={FUZZYWUZZY_AVAILABLE}, "
              f"python-Levenshtein={PYTHON_LEVENSHTEIN_AVAILABLE}")

        try:
            self.run_single_comparison_benchmarks()
            self.run_batch_processing_benchmarks()
            self.run_fuzzy_search_benchmarks()
            self.run_memory_efficiency_test()

            self.calculate_relative_speeds()
            self.print_summary()

        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        except Exception as e:
            print(f"\nBenchmark failed with error: {e}")
            traceback.print_exc()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="fast_fuzzy Performance Benchmark Suite")
    parser.add_argument("--size", type=int, default=5000,
                       help="Size of test dataset (default: 5000)")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations per test (default: 5)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="Output file for results (default: benchmark_results.json)")

    args = parser.parse_args()

    # Validate arguments
    if args.size < 100:
        print("ERROR: Dataset size must be at least 100")
        sys.exit(1)

    if args.iterations < 1:
        print("ERROR: Iterations must be at least 1")
        sys.exit(1)

    # Check if any libraries are available
    if not any([fast_fuzzy_AVAILABLE, RAPIDFUZZ_AVAILABLE, FUZZYWUZZY_AVAILABLE, PYTHON_LEVENSHTEIN_AVAILABLE]):
        print("ERROR: No fuzzy string matching libraries available for benchmarking")
        print("Please install at least one of: fast_fuzzy, rapidfuzz, fuzzywuzzy, python-Levenshtein")
        sys.exit(1)

    # Run benchmarks
    suite = BenchmarkSuite(size=args.size, iterations=args.iterations)
    suite.run_all_benchmarks()

    # Save results
    if args.output:
        suite.save_results(args.output)

if __name__ == "__main__":
    main()
