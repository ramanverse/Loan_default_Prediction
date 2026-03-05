"""
Tests for ensemble methods.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_voting_classifier_hard(sample_data):
    """Test hard voting classifier."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf1 = LogisticRegression(random_state=42)
    clf2 = DecisionTreeClassifier(random_state=42)
    
    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('dt', clf2)],
        voting='hard'
    )
    
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    
    assert len(y_pred) == len(y_test)
    assert all(pred in [0, 1] for pred in y_pred)
    assert accuracy_score(y_test, y_pred) > 0.5


def test_voting_classifier_soft(sample_data):
    """Test soft voting classifier."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf1 = LogisticRegression(random_state=42)
    clf2 = DecisionTreeClassifier(random_state=42)
    
    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('dt', clf2)],
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    y_proba = voting_clf.predict_proba(X_test)
    
    assert len(y_pred) == len(y_test)
    assert y_proba.shape == (len(y_test), 2)
    assert all(0 <= p <= 1 for p in y_proba.flatten())


def test_stacking_classifier(sample_data):
    """Test stacking classifier."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    base_estimators = [
        ('lr', LogisticRegression(random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42))
    ]
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=3
    )
    
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    y_proba = stacking_clf.predict_proba(X_test)
    
    assert len(y_pred) == len(y_test)
    assert y_proba.shape == (len(y_test), 2)
    assert accuracy_score(y_test, y_pred) > 0.5


def test_blending_predictions(sample_data):
    """Test blending predictions from multiple models."""
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf1 = LogisticRegression(random_state=42)
    clf2 = DecisionTreeClassifier(random_state=42)
    
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    
    y_proba1 = clf1.predict_proba(X_test)[:, 1]
    y_proba2 = clf2.predict_proba(X_test)[:, 1]
    
    # Blend with equal weights
    y_proba_blend = 0.5 * y_proba1 + 0.5 * y_proba2
    y_pred_blend = (y_proba_blend >= 0.5).astype(int)
    
    assert len(y_pred_blend) == len(y_test)
    assert all(0 <= p <= 1 for p in y_proba_blend)
    assert accuracy_score(y_test, y_pred_blend) > 0.3
