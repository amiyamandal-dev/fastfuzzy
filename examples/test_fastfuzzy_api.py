#!/usr/bin/env python3
"""
fastfuzzy API Test Script

This script checks what functions and classes are available in your fastfuzzy installation
and tests basic functionality.

Usage:
    python test_fastfuzzy_api.py
"""

def test_fastfuzzy_api():
    """Test what's available in the fastfuzzy module."""

    print("fastfuzzy API Test")
    print("=" * 50)

    # Try to import fastfuzzy
    try:
        import fastfuzzy
        print("✅ fastfuzzy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fastfuzzy: {e}")
        return

    # Check module attributes
    print(f"\n📋 Available attributes in fastfuzzy module:")
    attributes = [attr for attr in dir(fastfuzzy) if not attr.startswith('_')]
    for attr in sorted(attributes):
        attr_type = type(getattr(fastfuzzy, attr)).__name__
        print(f"  - {attr} ({attr_type})")

    # Test basic functionality
    print(f"\n🧪 Testing basic functionality:")

    test_strings = ("hello", "helo")

    # Test high-level API functions
    high_level_functions = [
        ('ratio', 'fastfuzzy.ratio'),
        ('partial_ratio', 'fastfuzzy.partial_ratio'),
        ('token_sort_ratio', 'fastfuzzy.token_sort_ratio'),
        ('levenshtein', 'fastfuzzy.levenshtein'),
        ('jaro_winkler', 'fastfuzzy.jaro_winkler'),
    ]

    for func_name, func_path in high_level_functions:
        try:
            func = eval(func_path)
            result = func(*test_strings, 0.0)
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
    ]

    for class_name in classes_to_test:
        try:
            cls = getattr(fastfuzzy, class_name)
            instance = cls()
            print(f"  ✅ {class_name} class available")

            # Test common methods
            if class_name == 'Levenshtein':
                if hasattr(instance, 'match_string_percentage'):
                    result = instance.match_string_percentage(*test_strings, 0.0)
                    print(f"    📊 match_string_percentage: {result}")

        except AttributeError:
            print(f"  ❌ {class_name} class not available")
        except Exception as e:
            print(f"  ⚠️  {class_name} error: {e}")

    # Test module-level objects
    print(f"\n🔗 Testing module-level objects:")

    module_objects = ['fuzz', 'process', 'utils', 'string_metric']

    for obj_name in module_objects:
        try:
            obj = getattr(fastfuzzy, obj_name)
            print(f"  ✅ {obj_name} object available ({type(obj).__name__})")

            if obj_name == 'fuzz' and hasattr(obj, 'ratio'):
                result = obj.ratio(*test_strings, 0.0)
                print(f"    📊 fuzz.ratio: {result}")

        except AttributeError:
            print(f"  ❌ {obj_name} object not available")
        except Exception as e:
            print(f"  ⚠️  {obj_name} error: {e}")

if __name__ == "__main__":
    test_fastfuzzy_api()
