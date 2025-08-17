#!/usr/bin/env python3
"""Minimal test to identify which function causes segmentation fault."""

print("Importing fastfuzzy...")
import fastfuzzy
print("✅ fastfuzzy imported successfully")

print("\nTesting basic instantiation...")
try:
    lev = fastfuzzy.Levenshtein()
    print("✅ Levenshtein instantiated successfully")
except Exception as e:
    print(f"❌ Levenshtein instantiation failed: {e}")

print("\nTesting cache functions...")
try:
    fastfuzzy.clear_global_cache()
    print("✅ clear_global_cache() executed successfully")
except Exception as e:
    print(f"❌ clear_global_cache() failed: {e}")

try:
    stats = fastfuzzy.global_cache_stats()
    print(f"✅ global_cache_stats() executed successfully: {stats}")
except Exception as e:
    print(f"❌ global_cache_stats() failed: {e}")

print("\nTesting simple function...")
try:
    result = fastfuzzy.ratio("hello", "helo")
    print(f"✅ ratio() executed successfully: {result}")
except Exception as e:
    print(f"❌ ratio() failed: {e}")