"""
Data validation utilities.
Validates input data for predictions and training.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class PredictionInput(BaseModel):
    """Validation schema for single prediction input."""
    
    amt_income_total: float = Field(..., ge=0, description="Total income")
    amt_credit: float = Field(..., ge=0, description="Credit amount")
    amt_annuity: float = Field(..., ge=0, description="Annuity amount")
    cnt_children: int = Field(0, ge=0, le=20, description="Number of children")
    ext_source_2: Optional[float] = Field(None, ge=0, le=1, description="External source 2")
    ext_source_3: Optional[float] = Field(None, ge=0, le=1, description="External source 3")
    age: float = Field(..., ge=18, le=100, description="Age")
    employ_years: float = Field(0.0, ge=0, le=50, description="Years of employment")
    
    @validator('amt_credit')
    def credit_should_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Credit amount must be positive')
        return v
    
    @validator('amt_income_total')
    def income_should_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Income must be positive')
        return v


class BatchPredictionInput(BaseModel):
    """Validation schema for batch prediction input."""
    
    records: List[Dict[str, Any]] = Field(..., min_items=1, max_items=10000)
    
    @validator('records')
    def validate_records(cls, v):
        if not v:
            raise ValueError('Records list cannot be empty')
        return v


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> tuple[bool, Optional[str]]:
    """
    Validate that DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty or None"
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {', '.join(missing_cols)}"
    
    return True, None


def validate_numeric_range(df: pd.DataFrame, column: str, min_val: float = None, max_val: float = None) -> tuple[bool, Optional[str]]:
    """
    Validate numeric column is within specified range.
    
    Args:
        df: DataFrame to validate
        column: Column name to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if column not in df.columns:
        return False, f"Column {column} not found"
    
    if min_val is not None and (df[column] < min_val).any():
        return False, f"Column {column} has values below minimum {min_val}"
    
    if max_val is not None and (df[column] > max_val).any():
        return False, f"Column {column} has values above maximum {max_val}"
    
    return True, None
