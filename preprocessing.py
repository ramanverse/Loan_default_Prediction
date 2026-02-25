"""
Preprocessing utilities for loan default prediction.
Extracted for reusability and testing.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_batch_data(batch_df, imputer=None, scaler=None, feature_columns=None):
    """
    Preprocess batch data for prediction using the same pipeline as training.
    
    Args:
        batch_df: DataFrame with raw input data
        imputer: Fitted SimpleImputer (if None, will fit new one)
        scaler: Fitted StandardScaler (if None, will fit new one)
        feature_columns: List of expected feature columns in order
    
    Returns:
        Tuple of (processed_df, imputer, scaler, feature_columns)
    """
    df_clean = batch_df.copy()
    
    # Drop columns with >40% missing values (same as training)
    missing = df_clean.isnull().mean()
    cols_to_drop = missing[missing > 0.40].index.tolist()
    df_clean = df_clean.drop(columns=cols_to_drop)
    
    # Handle anomalous DAYS_EMPLOYED
    if 'DAYS_EMPLOYED' in df_clean.columns:
        df_clean['DAYS_EMPLOYED'] = df_clean['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # Feature Engineering (same as training)
    if 'DAYS_BIRTH' in df_clean.columns:
        df_clean['AGE'] = -df_clean['DAYS_BIRTH'] / 365.0
    
    if 'DAYS_EMPLOYED' in df_clean.columns:
        df_clean['EMPLOY_YEARS'] = -(df_clean['DAYS_EMPLOYED'] / 365.0)
    
    if {'AMT_CREDIT', 'AMT_INCOME_TOTAL'}.issubset(df_clean.columns):
        df_clean['CREDIT_INCOME_RATIO'] = df_clean['AMT_CREDIT'] / df_clean['AMT_INCOME_TOTAL'].replace({0: np.nan})
    
    if {'AMT_ANNUITY', 'AMT_INCOME_TOTAL'}.issubset(df_clean.columns):
        df_clean['ANNUITY_INCOME_RATIO'] = df_clean['AMT_ANNUITY'] / df_clean['AMT_INCOME_TOTAL'].replace({0: np.nan})
    
    exts = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df_clean.columns]
    if exts:
        df_clean['EXT_SOURCES_MEAN'] = df_clean[exts].mean(axis=1)
    
    # Impute numeric columns
    num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols = [c for c in num_cols if c not in ('SK_ID_CURR', 'TARGET')]
    
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
    else:
        df_clean[num_cols] = imputer.transform(df_clean[num_cols])
    
    # Encode categorical columns
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        df_clean[c] = df_clean[c].fillna('MISSING')
        df_clean[c] = pd.Categorical(df_clean[c]).codes
    
    # IQR capping for financial variables
    def cap_iqr(series, k=1.5):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        return series.clip(lower=lower, upper=upper)
    
    cap_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_INCOME_RATIO']
    cap_features = [c for c in cap_features if c in df_clean.columns]
    for c in cap_features:
        df_clean[c] = cap_iqr(df_clean[c], k=1.5)
    
    # Align columns with training features
    if feature_columns is not None:
        # Add missing columns with 0
        missing_cols = set(feature_columns) - set(df_clean.columns)
        for col in missing_cols:
            df_clean[col] = 0
        
        # Remove extra columns
        extra_cols = set(df_clean.columns) - set(feature_columns)
        df_clean = df_clean.drop(columns=extra_cols)
        
        # Reorder to match training
        df_clean = df_clean[feature_columns]
    
    return df_clean, imputer, scaler, feature_columns
