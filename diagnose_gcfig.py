#!/usr/bin/env python3
"""
Diagnostic script to check GCf-IG computation for each method.
"""

import sys
sys.path.insert(0, '/home/n88r684/gcf_ig')

import numpy as np
import torch
from data_utils import load_german_credit
from baselines import train_baseline_method
from metrics import build_label_group_baselines, compute_gcf_ig_metric

# Load data
print("Loading German Credit dataset...")
train_data, test_data = load_german_credit('german.csv')
print(f"Train: {len(train_data.X)}, Test: {len(test_data.X)}")

# Build baselines
print("\nBuilding baselines from training data...")
baselines = build_label_group_baselines(train_data.X, train_data.y, train_data.a, method="mean")
print(f"Baselines keys: {list(baselines.keys())}")
for key, val in baselines.items():
    print(f"  {key}: shape={val.shape}, mean={val.mean():.4f}")

# Test config
config = {
    'hidden_dims': [64, 32],
    'n_epochs': 5,  # Just 5 epochs for quick test
    'batch_size': 64,
    'learning_rate': 0.001,
    'lambda_ig': 0.0,  # No GCf-IG loss during training
    'lambda_sp': 0.0,
    'device': 'cpu'
}

methods_to_test = ['unconstrained', 'lagrangian']

for method_name in methods_to_test:
    print(f"\n{'='*80}")
    print(f"Testing: {method_name}")
    print(f"{'='*80}")
    
    # Train
    print(f"Training {method_name}...")
    method, results = train_baseline_method(
        method_name,
        train_data,
        test_data,
        config,
        device='cpu',
        verbose=False
    )
    
    # Check model type
    print(f"\nModel type: {type(method)}")
    print(f"Has .model attr: {hasattr(method, 'model')}")
    if hasattr(method, 'model'):
        print(f"Inner model type: {type(method.model)}")
        print(f"Is PyTorch: {isinstance(method.model, torch.nn.Module)}")
    
    # Manually compute GCf-IG
    print(f"\n{'=== Manual GCf-IG Computation ==='}")
    gcf_ig_value, details = compute_gcf_ig_metric(
        method,
        test_data.X,
        test_data.y,
        test_data.a,
        baselines,
        n_steps=32,
        device='cpu'
    )
    
    print(f"GCf-IG value: {gcf_ig_value}")
    print(f"GCf-IG (scientific): {gcf_ig_value:.15e}")
    print(f"Is suspiciously small? {gcf_ig_value < 0.001}")
    
    # Check predictions
    print(f"\n{'=== Predictions Check ==='}")
    if hasattr(method, 'predict_proba'):
        probs = method.predict_proba(test_data.X[:10])
        print(f"Sample predictions (first 10): {probs[:10]}")
        print(f"Prediction range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    print(f"\n{'=== Reported Test Metrics ==='}")
    print(f"Test accuracy: {results['test'].get('accuracy', 'N/A')}")
    print(f"Test GCf-IG (from results): {results['test'].get('gcf_ig', 'N/A')}")

print(f"\n{'='*80}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*80}")
