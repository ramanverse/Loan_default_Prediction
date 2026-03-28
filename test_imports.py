#!/usr/bin/env python3
"""
Quick test script to verify all imports work correctly
"""
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

print("Testing imports...")
print(f"Python path: {sys.path[:3]}...")
print()

# Test basic imports
try:
    import streamlit as st
    print("✓ streamlit")
except Exception as e:
    print(f"✗ streamlit: {e}")

try:
    import pandas as pd
    print("✓ pandas")
except Exception as e:
    print(f"✗ pandas: {e}")

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

# Test utils imports
print("\nTesting utils imports...")
try:
    from utils.logger import setup_logger
    logger = setup_logger("test")
    print("✓ utils.logger")
except Exception as e:
    print(f"✗ utils.logger: {e}")

try:
    from utils.config import config
    print("✓ utils.config")
except Exception as e:
    print(f"✗ utils.config: {e}")

try:
    from utils.metrics import calculate_comprehensive_metrics, PerformanceMonitor
    print("✓ utils.metrics")
except Exception as e:
    print(f"✗ utils.metrics: {e}")

try:
    from utils.feature_selection import get_feature_importance_scores
    print("✓ utils.feature_selection")
except Exception as e:
    print(f"✗ utils.feature_selection: {e}")

print("\nAll critical imports tested!")
