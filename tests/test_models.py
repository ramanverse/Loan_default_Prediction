"""
Integration tests for model training and prediction.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from preprocessing import preprocess_batch_data


def test_model_training_pipeline():
    """Test complete model training pipeline."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.7).astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Validate predictions
    assert len(y_pred) == len(y_test)
    assert len(y_proba) == len(y_test)
    assert all(y_pred >= 0) and all(y_pred <= 1)
    assert all(y_proba >= 0) and all(y_proba <= 1)
    
    # Check performance
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.5  # Should be better than random
    
    if len(np.unique(y_test)) > 1:
        roc_auc = roc_auc_score(y_test, y_proba)
        assert roc_auc > 0.5


def test_model_prediction_consistency():
    """Test that model predictions are consistent."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (np.random.rand(100) > 0.7).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_scaled, y)
    
    # Predict same input twice
    pred1 = model.predict(X_scaled[:5])
    pred2 = model.predict(X_scaled[:5])
    
    assert np.array_equal(pred1, pred2)


def test_preprocessing_consistency():
    """Test that preprocessing produces consistent results."""
    data = {
        'AMT_INCOME_TOTAL': [200000, 300000, 150000],
        'AMT_CREDIT': [400000, 600000, 300000],
        'DAYS_BIRTH': [-10000, -12000, -9000],
        'TARGET': [0, 1, 0]
    }
    df = pd.DataFrame(data)
    
    processed1, _, _, _ = preprocess_batch_data(df)
    processed2, _, _, _ = preprocess_batch_data(df)
    
    # Check that AGE is calculated consistently
    if 'AGE' in processed1.columns and 'AGE' in processed2.columns:
        assert np.allclose(processed1['AGE'], processed2['AGE'])


def test_model_with_imbalanced_data():
    """Test model performance with imbalanced data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create imbalanced data (90% class 0, 10% class 1)
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples) > 0.9).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train with class weights
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=200)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Should still make predictions
    assert len(y_pred) == len(y_test)
    if len(np.unique(y_test)) > 1:
        assert roc_auc_score(y_test, y_proba) > 0.5


def test_multiple_models_comparison():
    """Test comparing multiple models."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    
    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=200),
        'tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'rf': RandomForestClassifier(n_estimators=10, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        if name == 'logistic':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
    
    # All models should perform better than random
    for name, metrics in results.items():
        assert metrics['roc_auc'] > 0.5
        assert metrics['accuracy'] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
