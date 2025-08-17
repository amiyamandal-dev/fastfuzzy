#!/usr/bin/env python3
"""
Test specific fastfuzzy functions to identify which ones work
"""

def test_functions():
    print("Testing specific fastfuzzy functions...")
    
    try:
        import fastfuzzy
        print("✅ fastfuzzy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fastfuzzy: {e}")
        return
    
    # Test simple attribute access
    simple_attrs = [
        'fuzz',
        'process',
        'utils',
        'string_metric'
    ]
    
    for attr_name in simple_attrs:
        try:
            attr = getattr(fastfuzzy, attr_name)
            print(f"✅ {attr_name} = {type(attr).__name__}")
        except Exception as e:
            print(f"❌ {attr_name} failed: {e}")
    
    # Test functions that should work without parameters
    simple_functions = [
        'clear_global_cache',
        'global_cache_stats',
    ]
    
    for func_name in simple_functions:
        try:
            func = getattr(fastfuzzy, func_name)
            if callable(func):
                result = func()
                print(f"✅ {func_name}() = {result}")
            else:
                print(f"✅ {func_name} = {func}")
        except Exception as e:
            print(f"❌ {func_name} failed: {e}")
    
    # Test basic functions with parameters
    test_cases = [
        ("hello", "helo"),
        ("world", "word"),
        ("test", "best")
    ]
    
    functions_to_test = [
        'ratio',
        'partial_ratio',
        'token_sort_ratio',
        'levenshtein',
        'jaro_winkler',
    ]
    
    print("\nTesting functions with parameters:")
    for func_name in functions_to_test:
        try:
            func = getattr(fastfuzzy, func_name)
            # Test with first pair
            result = func(test_cases[0][0], test_cases[0][1], 0.0)
            print(f"✅ {func_name}('{test_cases[0][0]}', '{test_cases[0][1]}', 0.0) = {result}")
        except Exception as e:
            print(f"❌ {func_name} failed: {e}")

if __name__ == "__main__":
    test_functions()
