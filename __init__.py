"""
GCf-IG: Group Counterfactual Integrated Gradients for Fair ML

A modular implementation for fairness experiments.
"""

from data_utils import FairnessDataset, load_german_credit, load_dataset
from models import MLPClassifier, create_model
from trainer import GCfIGTrainer, BaselineManager
from losses import GCfIGLoss, StatisticalParityLoss, EqualizedOddsLoss
from metrics import (
    compute_performance_metrics,
    compute_statistical_parity,
    compute_equalized_odds,
    compute_gcf_ig_metric,
    evaluate_model,
    print_metrics
)

__version__ = "0.2.0"

__all__ = [
    # Data
    'FairnessDataset',
    'load_german_credit',
    'load_dataset',
    
    # Models
    'MLPClassifier',
    'create_model',
    
    # Training
    'GCfIGTrainer',
    'BaselineManager',
    
    # Losses
    'GCfIGLoss',
    'StatisticalParityLoss',
    'EqualizedOddsLoss',
    
    # Metrics
    'compute_performance_metrics',
    'compute_statistical_parity',
    'compute_equalized_odds',
    'compute_gcf_ig_metric',
    'evaluate_model',
    'print_metrics',
]
