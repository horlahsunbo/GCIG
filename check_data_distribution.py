#!/usr/bin/env python3
"""Check data distribution in German Credit dataset"""

import sys
sys.path.insert(0, '/home/n88r684/gcf_ig')

import numpy as np
from data_utils import load_german_credit

# Load data
print("Loading German Credit dataset...")
train_data, test_data = load_german_credit('german.csv')

print(f"\n{'='*80}")
print("TRAINING DATA ANALYSIS")
print(f"{'='*80}")

# Check label distribution
print("\nLabel distribution:")
for label in [0, 1]:
    count = (train_data.y == label).sum()
    pct = count / len(train_data.y) * 100
    print(f"  Label {label}: {count} samples ({pct:.1f}%)")

# Check group distribution  
print("\nGroup distribution:")
for group in [0, 1]:
    count = (train_data.a == group).sum()
    pct = count / len(train_data.a) * 100
    print(f"  Group {group}: {count} samples ({pct:.1f}%)")

# Check (label, group) combinations
print("\nCombined (label, group) distribution:")
for label in [0, 1]:
    for group in [0, 1]:
        mask = (train_data.y == label) & (train_data.a == group)
        count = mask.sum()
        pct = count / len(train_data.y) * 100
        print(f"  (Label={label}, Group={group}): {count} samples ({pct:.1f}%)")
        
        if count > 0:
            # Check feature means for this group
            X_subset = train_data.X[mask]
            mean_vals = X_subset.mean(axis=0)
            print(f"    Feature mean: {mean_vals.mean():.6f}")
            print(f"    Feature std:  {mean_vals.std():.6f}")

# Check if groups are actually different
print(f"\n{'='*80}")
print("GROUP COMPARISON")
print(f"{'='*80}")

for label in [0, 1]:
    print(f"\nLabel {label}:")
    
    mask_g0 = (train_data.y == label) & (train_data.a == 0)
    mask_g1 = (train_data.y == label) & (train_data.a == 1)
    
    if mask_g0.sum() > 0 and mask_g1.sum() > 0:
        X_g0 = train_data.X[mask_g0]
        X_g1 = train_data.X[mask_g1]
        
        mean_g0 = X_g0.mean(axis=0)
        mean_g1 = X_g1.mean(axis=0)
        
        print(f"  Group 0 mean features: {mean_g0[:5]} ...")
        print(f"  Group 1 mean features: {mean_g1[:5]} ...")
        print(f"  Difference: {np.abs(mean_g0 - mean_g1)[:5]} ...")
        print(f"  Max difference: {np.abs(mean_g0 - mean_g1).max():.6f}")
        print(f"  Mean difference: {np.abs(mean_g0 - mean_g1).mean():.6f}")
        
        # Are they identical?
        if np.allclose(mean_g0, mean_g1, atol=1e-6):
            print(f"  ⚠️  WARNING: Group means are nearly IDENTICAL!")
        else:
            print(f"  ✅ Groups have different feature distributions")
    else:
        print(f"  ⚠️  One or both groups have no samples!")

print(f"\n{'='*80}")
print("DIAGNOSIS: Check if protected attribute 'a' is properly loaded")
print(f"{'='*80}")

print(f"\nUnique values in 'a': {np.unique(train_data.a)}")
print(f"Type of 'a': {type(train_data.a)}")
print(f"Dtype of 'a': {train_data.a.dtype}")
print(f"First 20 values of 'a': {train_data.a[:20]}")
