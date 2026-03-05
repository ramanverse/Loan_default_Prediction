"""
Feature selection utilities.
Provides methods for selecting important features.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Optional


def select_k_best_features(X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top k features using univariate statistical tests.
    
    Args:
        X: Feature matrix
        y: Target variable
        k: Number of features to select
    
    Returns:
        Tuple of (selected_features_df, selected_feature_names)
    """
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features


def select_features_by_importance(X: pd.DataFrame, y: pd.Series, 
                                  n_features: int = 20,
                                  threshold: Optional[float] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features using Random Forest importance.
    
    Args:
        X: Feature matrix
        y: Target variable
        n_features: Number of features to select
        threshold: Minimum importance threshold (optional)
    
    Returns:
        Tuple of (selected_features_df, selected_feature_names)
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    if threshold:
        selected_features = importances[importances >= threshold].index.tolist()
    else:
        selected_features = importances.head(n_features).index.tolist()
    
    return X[selected_features], selected_features


def get_feature_importance_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Get feature importance scores from multiple methods.
    
    Args:
        X: Feature matrix
        y: Target variable
    
    Returns:
        DataFrame with feature importance scores
    """
    scores = pd.DataFrame(index=X.columns)
    
    # F-test scores
    f_scores, _ = f_classif(X, y)
    scores['f_score'] = f_scores
    
    # Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    scores['mutual_info'] = mi_scores
    
    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    scores['rf_importance'] = rf.feature_importances_
    
    # Normalize scores
    for col in scores.columns:
        scores[col] = (scores[col] - scores[col].min()) / (scores[col].max() - scores[col].min() + 1e-8)
    
    scores['combined_score'] = scores.mean(axis=1)
    scores = scores.sort_values('combined_score', ascending=False)
    
    return scores
