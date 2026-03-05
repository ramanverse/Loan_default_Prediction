"""
Unit tests for preprocessing functions.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import preprocess_batch_data


def test_preprocess_batch_data_basic():
    """Test basic preprocessing functionality"""
    # Create sample data
    data = {
        'SK_ID_CURR': [1, 2, 3],
        'AMT_INCOME_TOTAL': [200000, 300000, 150000],
        'AMT_CREDIT': [400000, 600000, 300000],
        'DAYS_BIRTH': [-10000, -12000, -9000],
        'DAYS_EMPLOYED': [-2000, -3000, -1500],
        'TARGET': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    processed, imputer, scaler, features = preprocess_batch_data(df)
    
    # Check that AGE and EMPLOY_YEARS are created
    assert 'AGE' in processed.columns
    assert 'EMPLOY_YEARS' in processed.columns
    
    # Check that AGE is positive
    assert all(processed['AGE'] > 0)
    
    # Check that TARGET is removed from numeric columns
    assert 'TARGET' not in processed.select_dtypes(include=['int64', 'float64']).columns or 'TARGET' not in processed.columns


def test_preprocess_batch_data_missing_values():
    """Test handling of missing values"""
    data = {
        'SK_ID_CURR': [1, 2, 3],
        'AMT_INCOME_TOTAL': [200000, np.nan, 150000],
        'AMT_CREDIT': [400000, 600000, np.nan],
        'DAYS_BIRTH': [-10000, -12000, -9000],
        'TARGET': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    processed, imputer, scaler, features = preprocess_batch_data(df)
    
    # Check that missing values are imputed
    assert not processed['AMT_INCOME_TOTAL'].isnull().any()
    assert not processed['AMT_CREDIT'].isnull().any()


def test_preprocess_batch_data_feature_engineering():
    """Test feature engineering"""
    data = {
        'SK_ID_CURR': [1, 2],
        'AMT_INCOME_TOTAL': [200000, 300000],
        'AMT_CREDIT': [400000, 600000],
        'AMT_ANNUITY': [25000, 35000],
        'DAYS_BIRTH': [-10000, -12000],
        'TARGET': [0, 1]
    }
    df = pd.DataFrame(data)
    
    processed, imputer, scaler, features = preprocess_batch_data(df)
    
    # Check that engineered features are created
    assert 'CREDIT_INCOME_RATIO' in processed.columns
    assert 'ANNUITY_INCOME_RATIO' in processed.columns
    
    # Check that ratios are calculated correctly
    assert abs(processed.loc[0, 'CREDIT_INCOME_RATIO'] - 2.0) < 0.01


def test_preprocess_batch_data_advanced_features():
    """Test advanced feature engineering."""
    data = {
        'SK_ID_CURR': [1, 2, 3],
        'AMT_INCOME_TOTAL': [200000, 300000, 150000],
        'AMT_CREDIT': [400000, 600000, 300000],
        'AMT_ANNUITY': [25000, 35000, 20000],
        'DAYS_BIRTH': [-10000, -12000, -9000],
        'DAYS_EMPLOYED': [-2000, -3000, -1500],
        'EXT_SOURCE_1': [0.5, 0.6, 0.4],
        'EXT_SOURCE_2': [0.7, 0.8, 0.6],
        'EXT_SOURCE_3': [0.6, 0.7, 0.5],
        'TARGET': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    processed, imputer, scaler, features = preprocess_batch_data(df)
    
    # Check advanced features are created
    assert 'EXT_SOURCES_MEAN' in processed.columns
    assert 'EXT_SOURCES_STD' in processed.columns
    assert 'EXT_SOURCES_MAX' in processed.columns
    assert 'EXT_SOURCES_MIN' in processed.columns
    
    # Check polynomial features
    assert 'CREDIT_INCOME_RATIO_SQ' in processed.columns
    assert 'CREDIT_INCOME_RATIO_SQRT' in processed.columns
    
    # Check interaction features
    assert 'AGE_EMPLOY_INTERACTION' in processed.columns
    assert 'AGE_EMPLOY_RATIO' in processed.columns
    
    # Check log transformations
    assert 'AMT_INCOME_TOTAL_LOG' in processed.columns
    assert 'AMT_CREDIT_LOG' in processed.columns


def test_preprocess_batch_data_binning():
    """Test binning features."""
    data = {
        'SK_ID_CURR': [1, 2, 3, 4, 5],
        'AMT_INCOME_TOTAL': [50000, 100000, 200000, 300000, 500000],
        'DAYS_BIRTH': [-9000, -12000, -15000, -18000, -20000],
        'TARGET': [0, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    
    processed, imputer, scaler, features = preprocess_batch_data(df)
    
    # Check binning features exist
    assert 'AGE_BINNED' in processed.columns
    assert 'INCOME_BINNED' in processed.columns
    
    # Check binned values are valid
    assert all(0 <= val <= 4 for val in processed['AGE_BINNED'].dropna())
    assert all(0 <= val <= 3 for val in processed['INCOME_BINNED'].dropna())


def test_preprocess_batch_data_edge_cases():
    """Test edge cases in preprocessing."""
    # Test with all missing values in a column
    data = {
        'SK_ID_CURR': [1, 2, 3],
        'AMT_INCOME_TOTAL': [np.nan, np.nan, np.nan],
        'AMT_CREDIT': [400000, 600000, 300000],
        'DAYS_BIRTH': [-10000, -12000, -9000],
        'TARGET': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    processed, imputer, scaler, features = preprocess_batch_data(df)
    
    # Should handle all missing values
    assert not processed['AMT_INCOME_TOTAL'].isnull().any()
    
    # Test with zero values
    data_zero = {
        'SK_ID_CURR': [1, 2],
        'AMT_INCOME_TOTAL': [0, 100000],
        'AMT_CREDIT': [400000, 600000],
        'AMT_ANNUITY': [25000, 35000],
        'DAYS_BIRTH': [-10000, -12000],
        'TARGET': [0, 1]
    }
    df_zero = pd.DataFrame(data_zero)
    
    processed_zero, _, _, _ = preprocess_batch_data(df_zero)
    
    # Should handle zero values without errors
    assert len(processed_zero) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
