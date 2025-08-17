#!/usr/bin/env python3
"""
Fuzzy Matcher Demo - Practical Examples

This script demonstrates practical use cases for the fastfuzzy library.
"""

import fastfuzzy
import time
import random
import string

def demo_basic_string_matching():
    """Demonstrate basic string matching capabilities."""
    print("=== Basic String Matching ===")
    
    # Test strings
    test_cases = [
        ("hello world", "hello word"),
        ("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear"),
        ("this is a test", "this is a test!"),
        ("Python Programming", "python programming"),
        ("Smith", "Smyth")
    ]
    
    for s1, s2 in test_cases:
        print(f"\nComparing: '{s1}' vs '{s2}'")
        
        # Basic ratio
        ratio = fastfuzzy.ratio(s1, s2, 0.0)
        print(f"  Ratio: {ratio:.1f}%")
        
        # Partial ratio
        partial = fastfuzzy.partial_ratio(s1, s2, 0.0)
        print(f"  Partial Ratio: {partial:.1f}%")
        
        # Token sort ratio
        token_sort = fastfuzzy.token_sort_ratio(s1, s2, 0.0)
        print(f"  Token Sort Ratio: {token_sort:.1f}%")
        
        # Token set ratio
        token_set = fastfuzzy.token_set_ratio(s1, s2, 0.0)
        print(f"  Token Set Ratio: {token_set:.1f}%")
        
        # Weighted ratio
        wratio = fastfuzzy.wratio(s1, s2, 0.0)
        print(f"  Weighted Ratio: {wratio:.1f}%")

def demo_fuzzy_search():
    """Demonstrate fuzzy search capabilities."""
    print("\n=== Fuzzy Search ===")
    
    # Sample data
    choices = [
        "Python Programming",
        "Java Development",
        "C++ Programming",
        "JavaScript Development",
        "Ruby on Rails",
        "Go Programming",
        "Rust Development",
        "Swift Programming",
        "Kotlin Development",
        "TypeScript Programming"
    ]
    
    queries = [
        "python program",
        "java dev",
        "rust language"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        
        # Extract top 3 matches
        results = fastfuzzy.process.extract(query, choices, limit=3)
        for choice, score, index in results:
            print(f"  {score:5.1f}% - {choice}")

def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing ===")
    
    # Generate test data
    queries = ["python", "java", "rust", "go", "javascript"]
    targets = [
        "Python Programming", "Java Development", "Rust Language",
        "Go Programming", "JavaScript Framework", "C++ Development"
    ]
    
    print(f"Comparing {len(queries)} queries with {len(targets)} targets...")
    
    # Time the operation
    start_time = time.perf_counter()
    matrix = fastfuzzy.batch_ratio(queries, targets)
    end_time = time.perf_counter()
    
    print(f"Completed in {end_time - start_time:.4f} seconds")
    
    # Display results
    print("\nSimilarity Matrix:")
    print(" " * 12 + " ".join(f"{t[:12]:>12}" for t in targets))
    for i, query in enumerate(queries):
        row = " ".join(f"{matrix[i][j]:>12.1f}" for j in range(len(targets)))
        print(f"{query:>12} {row}")

def demo_phonetic_matching():
    """Demonstrate phonetic matching capabilities."""
    print("\n=== Phonetic Matching ===")
    
    # Test names that sound similar
    test_names = [
        ("Smith", "Smyth"),
        ("Johnson", "Johnsen"),
        ("Williams", "Wilhems"),
        ("Brown", "Braun")
    ]
    
    phonetic = fastfuzzy.PhoneticAlgorithms()
    
    for name1, name2 in test_names:
        print(f"\nMatching: '{name1}' vs '{name2}'")
        
        # Soundex
        soundex1 = phonetic.soundex(name1)
        soundex2 = phonetic.soundex(name2)
        soundex_match = soundex1 == soundex2
        print(f"  Soundex: {soundex1} vs {soundex2} - Match: {soundex_match}")
        
        # Metaphone
        metaphone1 = phonetic.metaphone(name1)
        metaphone2 = phonetic.metaphone(name2)
        metaphone_match = metaphone1 == metaphone2
        print(f"  Metaphone: {metaphone1} vs {metaphone2} - Match: {metaphone_match}")

def demo_performance_comparison():
    """Demonstrate performance benefits."""
    print("\n=== Performance Comparison ===")
    
    # Generate test data
    def generate_strings(count, min_len=5, max_len=20):
        strings = []
        for _ in range(count):
            length = random.randint(min_len, max_len)
            s = ''.join(random.choices(string.ascii_letters + ' ', k=length))
            strings.append(s.strip())
        return strings
    
    queries = generate_strings(100)
    targets = generate_strings(1000)
    
    print(f"Testing with {len(queries)} queries and {len(targets)} targets...")
    
    # Test single comparisons
    start_time = time.perf_counter()
    for q in queries[:10]:  # Test first 10 queries
        for t in targets[:10]:  # Test first 10 targets
            fastfuzzy.ratio(q, t, 0.0)
    single_time = time.perf_counter() - start_time
    
    # Test batch processing
    start_time = time.perf_counter()
    fastfuzzy.batch_ratio(queries[:10], targets[:10])
    batch_time = time.perf_counter() - start_time
    
    print(f"Single comparisons (100): {single_time:.4f} seconds")
    print(f"Batch processing (100): {batch_time:.4f} seconds")
    print(f"Performance improvement: {single_time/batch_time:.1f}x faster")

def demo_cache_benefits():
    """Demonstrate cache benefits."""
    print("\n=== Cache Benefits ===")
    
    # Create a Levenshtein instance to access cache stats
    lev = fastfuzzy.Levenshtein()
    
    # Clear cache first
    lev.clear_cache()
    
    # Generate test data with repeated strings
    test_strings = ["hello world"] * 1000 + ["fuzzy matching"] * 1000
    
    print("Testing with repeated strings to demonstrate caching...")
    
    start_time = time.perf_counter()
    for i, s1 in enumerate(test_strings[:1000]):
        for s2 in test_strings[:100]:
            lev.match_string_percentage(s1, s2, 0.0)
    end_time = time.perf_counter()
    
    # Check cache stats
    stats = lev.cache_stats()
    hit_rate = stats['hit_rate_percent']
    
    print(f"Processing time: {end_time - start_time:.4f} seconds")
    print(f"Cache hit rate: {hit_rate}%")
    print(f"Cache size: {stats['size']}/{stats['capacity']}")

def demo_custom_scorer():
    """Demonstrate creating a custom scorer."""
    print("\n=== Custom Scorer ===")
    
    class CustomMatcher:
        def __init__(self):
            self.fuzzy = fastfuzzy.FuzzyRatio()
            self.utils = fastfuzzy.Utils()
        
        def custom_score(self, s1, s2, weights=None):
            """Custom scoring function with weighted components."""
            if weights is None:
                weights = {
                    'ratio': 0.3,
                    'partial': 0.3,
                    'token_sort': 0.2,
                    'token_set': 0.2
                }
            
            # Calculate components
            ratio = self.fuzzy.ratio(s1, s2) or 0.0
            partial = self.fuzzy.partial_ratio(s1, s2) or 0.0
            token_sort = self.fuzzy.token_sort_ratio(s1, s2) or 0.0
            token_set = self.fuzzy.token_set_ratio(s1, s2) or 0.0
            
            # Weighted combination
            final_score = (
                ratio * weights['ratio'] +
                partial * weights['partial'] +
                token_sort * weights['token_sort'] +
                token_set * weights['token_set']
            )
            
            return {
                'final_score': final_score,
                'components': {
                    'ratio': ratio,
                    'partial': partial,
                    'token_sort': token_sort,
                    'token_set': token_set
                }
            }
    
    # Test custom scorer
    matcher = CustomMatcher()
    result = matcher.custom_score("hello world", "world hello")
    
    print(f"Custom score: {result['final_score']:.1f}%")
    print("Components:")
    for component, score in result['components'].items():
        print(f"  {component}: {score:.1f}%")

def main():
    """Run all demos."""
    print("FastFuzzy Library Demo")
    print("=" * 50)
    
    try:
        demo_basic_string_matching()
        demo_fuzzy_search()
        demo_batch_processing()
        demo_phonetic_matching()
        demo_performance_comparison()
        demo_cache_benefits()
        demo_custom_scorer()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()