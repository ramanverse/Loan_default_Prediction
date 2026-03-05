"""
Tests for data validation utilities.
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validators import (
    PredictionInput, BatchPredictionInput,
    validate_dataframe, validate_numeric_range
)


def test_prediction_input_validation():
    """Test PredictionInput validation."""
    # Valid input
    valid_input = PredictionInput(
        amt_income_total=200000.0,
        amt_credit=400000.0,
        amt_annuity=25000.0,
        age=35.0,
        employ_years=5.0
    )
    assert valid_input.amt_income_total == 200000.0
    
    # Invalid input - negative income
    with pytest.raises(Exception):
        PredictionInput(
            amt_income_total=-1000.0,
            amt_credit=400000.0,
            amt_annuity=25000.0,
            age=35.0,
            employ_years=5.0
        )


def test_validate_dataframe():
    """Test DataFrame validation."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })
    
    # Valid
    is_valid, error = validate_dataframe(df, ['col1', 'col2'])
    assert is_valid
    assert error is None
    
    # Missing column
    is_valid, error = validate_dataframe(df, ['col1', 'col2', 'col3'])
    assert not is_valid
    assert 'col3' in error


def test_validate_numeric_range():
    """Test numeric range validation."""
    df = pd.DataFrame({
        'age': [25, 30, 35, 40]
    })
    
    # Valid range
    is_valid, error = validate_numeric_range(df, 'age', min_val=18, max_val=100)
    assert is_valid
    
    # Invalid range
    is_valid, error = validate_numeric_range(df, 'age', min_val=50, max_val=100)
    assert not is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
