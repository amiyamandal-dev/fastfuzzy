#!/usr/bin/env python3
"""
Test specific fastfuzzy functions to identify which ones cause segmentation faults
"""

def test_specific_functions():
    """Test specific fastfuzzy functions."""
    
    print("Testing specific fastfuzzy functions...")
    
    try:
        import fastfuzzy
        print("✅ fastfuzzy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fastfuzzy: {e}")
        return
    
    # Test simple functions that don't require parameters
    print(f"\n🔧 Testing simple functions:")
    
    simple_functions = [
        'clear_global_cache',
        'global_cache_stats',
    ]
    
    for func_name in simple_functions:
        try:
            func = getattr(fastfuzzy, func_name)
            if callable(func):
                result = func()
                print(f"  ✅ {func_name}() = {result}")
            else:
                print(f"  ✅ {func_name} = {func}")
        except Exception as e:
            print(f"  ❌ {func_name} failed: {e}")
    
    # Test class methods
    print(f"\n🏗️  Testing class methods:")
    
    # Test Levenshtein class
    try:
        lev = fastfuzzy.Levenshtein()
        print("  ✅ Levenshtein instantiated")
        
        # Test methods that might work
        methods_to_test = [
            'cache_stats',
            'clear_cache',
        ]
        
        for method_name in methods_to_test:
            try:
                method = getattr(lev, method_name)
                if callable(method):
                    result = method()
                    print(f"    ✅ lev.{method_name}() = {result}")
                else:
                    print(f"    ✅ lev.{method_name} = {method}")
            except Exception as e:
                print(f"    ❌ lev.{method_name} failed: {e}")
                
    except Exception as e:
        print(f"  ❌ Levenshtein failed: {e}")

if __name__ == "__main__":
    test_specific_functions()