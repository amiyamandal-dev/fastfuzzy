#!/usr/bin/env python3
"""
Simple test to check if fastfuzzy is working
"""

try:
    import fastfuzzy
    print("fastfuzzy imported successfully")
    
    # Test a simple function
    result = fastfuzzy.ratio("hello", "helo", 0.0)
    print(f"Ratio test: {result}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()