#!/usr/bin/env python3
"""
fast_fuzzy API Test Script

This script checks what functions and classes are available in your fast_fuzzy installation
and tests basic functionality.

Usage:
    python test_fast_fuzzy_api.py
"""

def test_fast_fuzzy_api():
    """Test what's available in the fast_fuzzy module."""

    print("fast_fuzzy API Test")
    print("=" * 50)

    # Try to import fast_fuzzy
    try:
        import fast_fuzzy
        print("✅ fast_fuzzy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fast_fuzzy: {e}")
        return

    # Check module attributes
    print(f"\n📋 Available attributes in fast_fuzzy module:")
    attributes = [attr for attr in dir(fast_fuzzy) if not attr.startswith('_')]
    for attr in sorted(attributes):
        attr_type = type(getattr(fast_fuzzy, attr)).__name__
        print(f"  - {attr} ({attr_type})")

    # Test basic functionality
    print(f"\n🧪 Testing basic functionality:")

    test_strings = ("hello", "helo")

    # Test high-level API functions
    high_level_functions = [
        ('ratio', 'fast_fuzzy.ratio'),
        ('partial_ratio', 'fast_fuzzy.partial_ratio'),
        ('token_sort_ratio', 'fast_fuzzy.token_sort_ratio'),
        ('levenshtein', 'fast_fuzzy.levenshtein'),
        ('jaro_winkler', 'fast_fuzzy.jaro_winkler'),
        ('quick_ratio', 'fast_fuzzy.quick_ratio'),
    ]

    for func_name, func_path in high_level_functions:
        try:
            func = eval(func_path)
            result = func(*test_strings)
            print(f"  ✅ {func_path}({test_strings[0]!r}, {test_strings[1]!r}) = {result}")
        except (AttributeError, NameError):
            print(f"  ❌ {func_path} not available")
        except Exception as e:
            print(f"  ⚠️  {func_path} error: {e}")

    # Test standalone functions
    standalone_functions = [
        ('match_string_percentage_fn', 'fast_fuzzy.match_string_percentage_fn'),
        ('match_string_percentage_list_fn', 'fast_fuzzy.match_string_percentage_list_fn'),
        ('batch_match_percentage', 'fast_fuzzy.batch_match_percentage'),
        ('batch_quick_ratio', 'fast_fuzzy.batch_quick_ratio'),
    ]

    print(f"\n🔧 Testing standalone functions:")
    for func_name, func_path in standalone_functions:
        try:
            func = eval(func_path)
            if 'batch' in func_name:
                result = func([test_strings[0]], [test_strings[1]])
                print(f"  ✅ {func_path}([{test_strings[0]!r}], [{test_strings[1]!r}]) = {result}")
            elif 'list' in func_name:
                result = func(test_strings[0], [test_strings[1]])
                print(f"  ✅ {func_path}({test_strings[0]!r}, [{test_strings[1]!r}]) = {result}")
            else:
                result = func(*test_strings)
                print(f"  ✅ {func_path}({test_strings[0]!r}, {test_strings[1]!r}) = {result}")
        except (AttributeError, NameError):
            print(f"  ❌ {func_path} not available")
        except Exception as e:
            print(f"  ⚠️  {func_path} error: {e}")

    # Test class-based API
    print(f"\n🏗️  Testing class-based API:")

    classes_to_test = [
        'Levenshtein',
        'Damerau',
        'StringMatcher',
        'FuzzyRatio',
        'Process',
        'Utils',
    ]

    for class_name in classes_to_test:
        try:
            cls = getattr(fast_fuzzy, class_name)
            instance = cls()
            print(f"  ✅ {class_name} class available")

            # Test common methods
            if class_name == 'Levenshtein':
                if hasattr(instance, 'match_string_percentage'):
                    result = instance.match_string_percentage(*test_strings)
                    print(f"    📊 match_string_percentage: {result}")
                if hasattr(instance, 'match_string_difference'):
                    result = instance.match_string_difference(*test_strings)
                    print(f"    📊 match_string_difference: {result}")

            elif class_name == 'StringMatcher':
                if hasattr(instance, 'jaro_winkler_difference'):
                    result = instance.jaro_winkler_difference(*test_strings)
                    print(f"    📊 jaro_winkler_difference: {result}")
                if hasattr(instance, 'normalized_levenshtein_difference'):
                    result = instance.normalized_levenshtein_difference(*test_strings)
                    print(f"    📊 normalized_levenshtein_difference: {result}")

            elif class_name == 'FuzzyRatio':
                if hasattr(instance, 'ratio'):
                    result = instance.ratio(*test_strings)
                    print(f"    📊 ratio: {result}")
                if hasattr(instance, 'partial_ratio'):
                    result = instance.partial_ratio(*test_strings)
                    print(f"    📊 partial_ratio: {result}")

        except AttributeError:
            print(f"  ❌ {class_name} class not available")
        except Exception as e:
            print(f"  ⚠️  {class_name} error: {e}")

    # Test module-level objects
    print(f"\n🔗 Testing module-level objects:")

    module_objects = ['fuzz', 'process', 'utils', 'string_metric']

    for obj_name in module_objects:
        try:
            obj = getattr(fast_fuzzy, obj_name)
            print(f"  ✅ {obj_name} object available ({type(obj).__name__})")

            if obj_name == 'fuzz' and hasattr(obj, 'ratio'):
                result = obj.ratio(*test_strings)
                print(f"    📊 fuzz.ratio: {result}")

            if obj_name == 'process' and hasattr(obj, 'extractOne'):
                choices = ["hello", "world", "helo", "help"]
                result = obj.extractOne(test_strings[0], choices)
                print(f"    📊 process.extractOne: {result}")

        except AttributeError:
            print(f"  ❌ {obj_name} object not available")
        except Exception as e:
            print(f"  ⚠️  {obj_name} error: {e}")

    # Performance test
    print(f"\n⚡ Quick performance test:")

    import time

    # Generate test data
    test_data = [f"test_string_{i}" for i in range(1000)]
    query = "test_string_500"

    # Test available functions
    performance_tests = []

    # Try to find a working similarity function
    if hasattr(fast_fuzzy, 'quick_ratio'):
        func = fast_fuzzy.quick_ratio
        performance_tests.append(('quick_ratio', func))
    elif hasattr(fast_fuzzy, 'Levenshtein'):
        lev = fast_fuzzy.Levenshtein()
        if hasattr(lev, 'match_string_percentage'):
            performance_tests.append(('Levenshtein.match_string_percentage', lev.match_string_percentage))

    for test_name, test_func in performance_tests:
        try:
            start_time = time.perf_counter()

            # Run 100 comparisons
            for target in test_data[:100]:
                test_func(query, target)

            end_time = time.perf_counter()
            duration = end_time - start_time
            ops_per_sec = 100 / duration

            print(f"  ✅ {test_name}: {ops_per_sec:.0f} ops/sec")

        except Exception as e:
            print(f"  ⚠️  {test_name} performance test failed: {e}")

    print(f"\n✨ fast_fuzzy API test completed!")

def test_benchmark_compatibility():
    """Test if fast_fuzzy can be used with the benchmark script."""

    print(f"\n🏁 Testing benchmark compatibility:")

    try:
        import fast_fuzzy

        # Test what the benchmark script expects
        test_pairs = [("hello", "helo"), ("world", "word"), ("test", "best")]

        # Functions the benchmark expects
        expected_functions = [
            'ratio', 'partial_ratio', 'token_sort_ratio',
            'levenshtein', 'jaro_winkler'
        ]

        working_functions = []

        for func_name in expected_functions:
            if hasattr(fast_fuzzy, func_name):
                try:
                    func = getattr(fast_fuzzy, func_name)
                    # Test with first pair
                    result = func(test_pairs[0][0], test_pairs[0][1])
                    working_functions.append(func_name)
                    print(f"  ✅ {func_name} works: {result}")
                except Exception as e:
                    try:
                        func = getattr(fast_fuzzy, func_name)
                        # Test with first pair
                        result = func(test_pairs[0][0], test_pairs[0][1],score_cutoff=0.0)
                        working_functions.append(func_name)
                        print(f"  ✅ {func_name} works: {result}")
                    except Exception as e:
                        print(f"  ⚠️  {func_name} exists but failed: {e}")
            else:
                print(f"  ❌ {func_name} not found")

        # Test alternative paths
        print(f"\n🔄 Testing alternative function paths:")

        # Test class-based approaches
        alternatives = []

        if hasattr(fast_fuzzy, 'Levenshtein'):
            lev = fast_fuzzy.Levenshtein()
            if hasattr(lev, 'match_string_percentage'):
                alternatives.append(('normalized_levenshtein', lambda s1, s2: lev.match_string_percentage(s1, s2)))
            if hasattr(lev, 'match_string_difference'):
                alternatives.append(('levenshtein_distance', lambda s1, s2: lev.match_string_difference(s1, s2)))

        if hasattr(fast_fuzzy, 'StringMatcher'):
            matcher = fast_fuzzy.StringMatcher()
            if hasattr(matcher, 'jaro_winkler_difference'):
                alternatives.append(('jaro_winkler_alt', lambda s1, s2: matcher.jaro_winkler_difference(s1, s2)))
            if hasattr(matcher, 'normalized_levenshtein_difference'):
                alternatives.append(('normalized_levenshtein_alt', lambda s1, s2: matcher.normalized_levenshtein_difference(s1, s2)))

        if hasattr(fast_fuzzy, 'FuzzyRatio'):
            fuzzy = fast_fuzzy.FuzzyRatio()
            if hasattr(fuzzy, 'ratio'):
                alternatives.append(('ratio_alt', lambda s1, s2: fuzzy.ratio(s1, s2)))

        for alt_name, alt_func in alternatives:
            try:
                result = alt_func(test_pairs[0][0], test_pairs[0][1])
                print(f"  ✅ {alt_name} alternative works: {result}")
            except Exception as e:
                print(f"  ⚠️  {alt_name} alternative failed: {e}")

        # Test batch functions
        print(f"\n📦 Testing batch functions:")

        batch_functions = [
            'batch_ratio', 'batch_levenshtein', 'batch_quick_ratio', 'batch_match_percentage'
        ]

        queries = [test_pairs[0][0]]
        targets = [test_pairs[0][1]]

        for batch_func_name in batch_functions:
            if hasattr(fast_fuzzy, batch_func_name):
                try:
                    batch_func = getattr(fast_fuzzy, batch_func_name)
                    result = batch_func(queries, targets)
                    print(f"  ✅ {batch_func_name} works: {result}")
                except Exception as e:
                    print(f"  ⚠️  {batch_func_name} failed: {e}")
            else:
                print(f"  ❌ {batch_func_name} not found")

        print(f"\n💡 Recommendations for benchmark script:")
        if working_functions:
            print(f"  - These direct functions work: {', '.join(working_functions)}")
        if alternatives:
            print(f"  - These alternatives work: {', '.join([name for name, _ in alternatives])}")

        return len(working_functions) > 0 or len(alternatives) > 0

    except ImportError:
        print(f"  ❌ fast_fuzzy not available")
        return False

if __name__ == "__main__":
    test_fast_fuzzy_api()
    compatible = test_benchmark_compatibility()

    if compatible:
        print(f"\n🎉 fast_fuzzy is ready for benchmarking!")
    else:
        print(f"\n⚠️  fast_fuzzy needs API adjustments for benchmarking")
