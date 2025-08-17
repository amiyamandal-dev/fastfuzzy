#!/usr/bin/env python3
"""
Minimal test to identify which fastfuzzy functions work
"""

def test_fastfuzzy_functions():
    """Test fastfuzzy functions one by one to identify segmentation faults."""
    
    print("Testing fastfuzzy functions...")
    
    try:
        import fastfuzzy
        print("✅ fastfuzzy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fastfuzzy: {e}")
        return
    
    # Test attributes
    print(f"\n📋 Available attributes in fastfuzzy module:")
    attributes = [attr for attr in dir(fastfuzzy) if not attr.startswith('_')]
    for attr in sorted(attributes):
        attr_type = type(getattr(fastfuzzy, attr)).__name__
        print(f"  - {attr} ({attr_type})")
    
    # Test basic instantiation of classes (without calling methods)
    print(f"\n🏗️  Testing class instantiation:")
    
    classes_to_test = [
        'Levenshtein',
        'Damerau',
        'StringMatcher',
        'FuzzyRatio',
        'Process',
        'Utils',
    ]
    
    instances = {}
    for class_name in classes_to_test:
        try:
            cls = getattr(fastfuzzy, class_name)
            instance = cls()
            instances[class_name] = instance
            print(f"  ✅ {class_name} instantiated successfully")
        except Exception as e:
            print(f"  ❌ {class_name} instantiation failed: {e}")
    
    # Test module-level objects
    print(f"\n🔗 Testing module-level objects:")
    
    module_objects = ['fuzz', 'process', 'utils', 'string_metric']
    
    for obj_name in module_objects:
        try:
            obj = getattr(fastfuzzy, obj_name)
            print(f"  ✅ {obj_name} object available ({type(obj).__name__})")
        except AttributeError:
            print(f"  ❌ {obj_name} object not available")
        except Exception as e:
            print(f"  ⚠️  {obj_name} error: {e}")

if __name__ == "__main__":
    test_fastfuzzy_functions()