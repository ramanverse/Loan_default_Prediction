"""
Performance metrics and monitoring utilities.
Tracks model performance and system metrics.
"""
import time
import functools
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics_history: list = []
        self.prediction_times: list = []
    
    def record_prediction_time(self, elapsed_time: float):
        """Record time taken for a prediction."""
        self.prediction_times.append(elapsed_time)
    
    def get_avg_prediction_time(self) -> float:
        """Get average prediction time."""
        if not self.prediction_times:
            return 0.0
        return sum(self.prediction_times) / len(self.prediction_times)
    
    def record_model_metrics(self, metrics: Dict[str, float]):
        """Record model performance metrics."""
        metrics['timestamp'] = time.time()
        self.metrics_history.append(metrics)
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """Get most recent metrics."""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1]
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of all recorded metrics."""
        if not self.metrics_history:
            return pd.DataFrame()
        return pd.DataFrame(self.metrics_history)


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        if hasattr(wrapper, 'monitor'):
            wrapper.monitor.record_prediction_time(elapsed_time)
        return result
    return wrapper
