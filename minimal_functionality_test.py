#!/usr/bin/env python3
"""
Minimal test to check if we can at least import and instantiate classes
"""

def minimal_test():
    print("Testing minimal fastfuzzy functionality...")
    
    try:
        import fastfuzzy
        print("✅ fastfuzzy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import fastfuzzy: {e}")
        return False
    
    # Test class instantiation
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
            print(f"✅ {class_name} instantiated successfully")
        except Exception as e:
            print(f"❌ {class_name} instantiation failed: {e}")
            return False
    
    # Test module-level objects
    module_objects = ['fuzz', 'process', 'utils', 'string_metric']
    
    for obj_name in module_objects:
        try:
            obj = getattr(fastfuzzy, obj_name)
            print(f"✅ {obj_name} object available ({type(obj).__name__})")
        except AttributeError:
            print(f"❌ {obj_name} object not available")
            return False
        except Exception as e:
            print(f"⚠️  {obj_name} error: {e}")
            return False
    
    print("🎉 All basic tests passed! Library is importable and instantiable.")
    return True

if __name__ == "__main__":
    success = minimal_test()
    if success:
        print("\n✅ Minimal test completed successfully")
    else:
        print("\n❌ Minimal test failed")