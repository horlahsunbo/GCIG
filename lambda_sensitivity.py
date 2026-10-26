#!/usr/bin/env python3
"""
Lambda Sensitivity Analysis for GCf-IG Method

This script varies lambda_ig and lambda_fair independently to analyze
their effect on utility, outcome fairness, and procedural fairness.

Usage:
    python lambda_sensitivity.py --output_dir sensitivity_results
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import torch

# Add your project path if needed
# sys.path.insert(0, '/home/n88r684/gcf_ig')

from data_utils import FairnessDataset
from additional_datasets import load_compas_data
from trainer import GCfIGTrainer
from models import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def run_single_experiment(lambda_ig, lambda_fair, train_data, test_data, 
                          base_params, device='cuda'):
    """
    Run a single training experiment with given lambda values.
    
    Args:
        lambda_ig: Weight for GCf-IG loss
        lambda_fair: Weight for fairness (EO) loss
        train_data: Training dataset
        test_data: Test dataset
        base_params: Dictionary of other hyperparameters
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary of test metrics
    """
    print(f"\n  Running: λ_ig={lambda_ig:.1f}, λ_fair={lambda_fair:.1f}")
    
    # Import here to avoid issues
    #from models import MLP
    
    # Create model
    model = MLPClassifier(
        input_dim=train_data.X.shape[1],
        hidden_dims=base_params['hidden_dims'],
        dropout=base_params.get('dropout', 0.2)
    )
    
    # Create config for trainer
    config = {
        'learning_rate': base_params['learning_rate'],
        'n_epochs': base_params['n_epochs'],
        'batch_size': base_params['batch_size'],
        'lambda_ig': lambda_ig,
        'lambda_eo': lambda_fair,
        'lambda_sp': 0.0,  # We're using EO, not SP
        'ig_steps': base_params.get('ig_steps', 32),
        'ema_rate': base_params.get('ema_rate', 0.1),
        'norm': base_params.get('norm', 'l2')
    }
    
    # Initialize trainer with model and config
    trainer = GCfIGTrainer(
        model=model,
        config=config,
        device=device
    )
    
    # Train
    trainer.train(
        train_data=train_data,
        val_data=None,
        verbose=False
    )
    
    # Evaluate on test set
    # Build baselines for GCf-IG computation
    from metrics import build_label_group_baselines
    baselines = build_label_group_baselines(
        train_data.X, 
        train_data.y, 
        train_data.a
    )
    
    # Get predictions
    y_pred = trainer.predict(test_data.X)
    
    # Compute metrics manually
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from metrics import compute_equalized_odds, compute_statistical_parity, compute_gcf_ig_metric
    
    y_pred_binary = (y_pred >= 0.5).astype(int).flatten()
    y_pred_probs = y_pred.flatten()
    
    # Standard metrics
    auc = float(roc_auc_score(test_data.y, y_pred_probs))
    f1 = float(f1_score(test_data.y, y_pred_binary))
    
    # Fairness metrics (these return tuples: (value, details))
    eo_gap, eo_details = compute_equalized_odds(test_data.y, y_pred_probs, test_data.a)
    sp_gap, sp_details = compute_statistical_parity(y_pred_probs, test_data.a)
    
    # GCf-IG score - use the existing function!
    gcf_ig_score, gcf_ig_details = compute_gcf_ig_metric(
        model=trainer,  # Pass the trainer (it has .model attribute)
        X=test_data.X,
        y=test_data.y,
        a=test_data.a,
        baselines=baselines,
        n_steps=32,
        norm='l2',
        device=device
    )
    
    test_metrics = {
        'auc': float(auc),
        'f1': float(f1),
        'equalized_odds': float(eo_gap),
        'statistical_parity': float(sp_gap),
        'gcf_ig': float(gcf_ig_score)
    }
    
    print(f"    F1: {test_metrics['f1']:.4f}, "
          f"EO: {test_metrics['equalized_odds']:.4f}, "
          f"GCf-IG: {test_metrics['gcf_ig']:.4f}")
    
    return test_metrics


def vary_lambda_ig(lambda_ig_values, lambda_fair_fixed, train_data, test_data,
                   base_params, device='cuda'):
    """
    Vary lambda_ig while keeping lambda_fair fixed.
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 1: Varying λ_ig (λ_fair fixed at {lambda_fair_fixed})")
    print("="*80)
    
    results = []
    
    for lambda_ig in lambda_ig_values:
        metrics = run_single_experiment(
            lambda_ig=lambda_ig,
            lambda_fair=lambda_fair_fixed,
            train_data=train_data,
            test_data=test_data,
            base_params=base_params,
            device=device
        )
        
        results.append({
            'lambda_ig': lambda_ig,
            'lambda_fair': lambda_fair_fixed,
            'experiment': 'vary_lambda_ig',
            **metrics
        })
    
    return pd.DataFrame(results)


def vary_lambda_fair(lambda_fair_values, lambda_ig_fixed, train_data, test_data,
                     base_params, device='cuda'):
    """
    Vary lambda_fair while keeping lambda_ig fixed.
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 2: Varying λ_fair (λ_ig fixed at {lambda_ig_fixed})")
    print("="*80)
    
    results = []
    
    for lambda_fair in lambda_fair_values:
        metrics = run_single_experiment(
            lambda_ig=lambda_ig_fixed,
            lambda_fair=lambda_fair,
            train_data=train_data,
            test_data=test_data,
            base_params=base_params,
            device=device
        )
        
        results.append({
            'lambda_ig': lambda_ig_fixed,
            'lambda_fair': lambda_fair,
            'experiment': 'vary_lambda_fair',
            **metrics
        })
    
    return pd.DataFrame(results)


def plot_sensitivity_analysis(df_ig, df_fair, tuned_lambda_ig, tuned_lambda_fair,
                               output_path):
    """
    Create sensitivity analysis plots.
    
    Args:
        df_ig: Results from varying lambda_ig
        df_fair: Results from varying lambda_fair
        tuned_lambda_ig: Your tuned lambda_ig value
        tuned_lambda_fair: Your tuned lambda_fair value
        output_path: Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Set style
    sns.set_style("whitegrid")
    colors = {'F1': '#2E86AB', 'EO Gap': '#A23B72', 'GCf-IG': "#F101A9"}
    
    # ========================================================================
    # Left plot: Varying lambda_ig
    # ========================================================================
    ax = axes[0]
    
    # Plot metrics
    ax.plot(df_ig['lambda_ig'], df_ig['f1'], 
            'o-', color=colors['F1'], linewidth=2.5, markersize=8, label='F1')
    ax.plot(df_ig['lambda_ig'], df_ig['equalized_odds'], 
            's-', color=colors['EO Gap'], linewidth=2.5, markersize=8, label='EO Gap')
    ax.plot(df_ig['lambda_ig'], df_ig['gcf_ig'], 
            '^-', color=colors['GCf-IG'], linewidth=2.5, markersize=8, label='GCIG')
    
    # Mark tuned value
    ax.axvline(x=tuned_lambda_ig, linestyle='--', color='gray', 
               linewidth=2, alpha=0.7, label=f'Tuned (λ_ig={tuned_lambda_ig})')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('λ_ig (Procedural Fairness Weight)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Varying λ_ig (λ_fair = {df_ig["lambda_fair"].iloc[0]})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # ========================================================================
    # Right plot: Varying lambda_fair
    # ========================================================================
    ax = axes[1]
    
    # Plot metrics
    ax.plot(df_fair['lambda_fair'], df_fair['f1'], 
            'o-', color=colors['F1'], linewidth=2.5, markersize=8, label='F1')
    ax.plot(df_fair['lambda_fair'], df_fair['equalized_odds'], 
            's-', color=colors['EO Gap'], linewidth=2.5, markersize=8, label='EO Gap')
    ax.plot(df_fair['lambda_fair'], df_fair['gcf_ig'], 
            '^-', color=colors['GCf-IG'], linewidth=2.5, markersize=8, label='GCIG')
    
    # Mark tuned value
    ax.axvline(x=tuned_lambda_fair, linestyle='--', color='gray', 
               linewidth=2, alpha=0.7, label=f'Tuned (λ_fair={tuned_lambda_fair})')
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('λ_fair (Outcome Fairness Weight)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Varying λ_fair (λ_ig = {df_fair["lambda_ig"].iloc[0]})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    
    # Overall title
    fig.suptitle('Hyperparameter Sensitivity Analysis (COMPAS Dataset)', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot to: {output_path}")
    
    return fig


def print_summary(df_ig, df_fair, tuned_lambda_ig, tuned_lambda_fair):
    """
    Print summary statistics.
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print("\n1. Varying λ_ig:")
    print(f"   λ_ig range: {df_ig['lambda_ig'].min()} - {df_ig['lambda_ig'].max()}")
    print(f"   F1 range:   {df_ig['f1'].min():.4f} - {df_ig['f1'].max():.4f} "
          f"(Δ = {(df_ig['f1'].max() - df_ig['f1'].min()):.4f})")
    print(f"   EO range:   {df_ig['equalized_odds'].min():.4f} - {df_ig['equalized_odds'].max():.4f} "
          f"(Δ = {(df_ig['equalized_odds'].max() - df_ig['equalized_odds'].min()):.4f})")
    print(f"   GCf-IG range: {df_ig['gcf_ig'].min():.4f} - {df_ig['gcf_ig'].max():.4f} "
          f"(Δ = {(df_ig['gcf_ig'].max() - df_ig['gcf_ig'].min()):.4f})")
    
    print("\n2. Varying λ_fair:")
    print(f"   λ_fair range: {df_fair['lambda_fair'].min()} - {df_fair['lambda_fair'].max()}")
    print(f"   F1 range:   {df_fair['f1'].min():.4f} - {df_fair['f1'].max():.4f} "
          f"(Δ = {(df_fair['f1'].max() - df_fair['f1'].min()):.4f})")
    print(f"   EO range:   {df_fair['equalized_odds'].min():.4f} - {df_fair['equalized_odds'].max():.4f} "
          f"(Δ = {(df_fair['equalized_odds'].max() - df_fair['equalized_odds'].min()):.4f})")
    print(f"   GCf-IG range: {df_fair['gcf_ig'].min():.4f} - {df_fair['gcf_ig'].max():.4f} "
          f"(Δ = {(df_fair['gcf_ig'].max() - df_fair['gcf_ig'].min()):.4f})")
    
    print("\n3. At Tuned Values:")
    # Find rows closest to tuned values
    idx_ig = (df_ig['lambda_ig'] - tuned_lambda_ig).abs().argmin()
    idx_fair = (df_fair['lambda_fair'] - tuned_lambda_fair).abs().argmin()
    
    print(f"   λ_ig = {tuned_lambda_ig}:")
    print(f"     F1: {df_ig.iloc[idx_ig]['f1']:.4f}")
    print(f"     EO: {df_ig.iloc[idx_ig]['equalized_odds']:.4f}")
    print(f"     GCf-IG: {df_ig.iloc[idx_ig]['gcf_ig']:.4f}")
    
    print(f"\n   λ_fair = {tuned_lambda_fair}:")
    print(f"     F1: {df_fair.iloc[idx_fair]['f1']:.4f}")
    print(f"     EO: {df_fair.iloc[idx_fair]['equalized_odds']:.4f}")
    print(f"     GCf-IG: {df_fair.iloc[idx_fair]['gcf_ig']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Lambda Sensitivity Analysis')
    parser.add_argument('--output_dir', type=str, default='sensitivity_results',
                       help='Directory to save results')
    parser.add_argument('--lambda_ig_values', type=float, nargs='+',
                       default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                       help='Lambda_ig values to test')
    parser.add_argument('--lambda_fair_values', type=float, nargs='+',
                       default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                       help='Lambda_fair values to test')
    parser.add_argument('--tuned_lambda_ig', type=float, default=1.0,
                       help='Your tuned lambda_ig value')
    parser.add_argument('--tuned_lambda_fair', type=float, default=1.0,
                       help='Your tuned lambda_fair value')
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LAMBDA SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Lambda_ig values: {args.lambda_ig_values}")
    print(f"Lambda_fair values: {args.lambda_fair_values}")
    print(f"Tuned values: λ_ig={args.tuned_lambda_ig}, λ_fair={args.tuned_lambda_fair}")
    print(f"Device: {args.device}")
    
    # Load data
    print("\n" + "="*80)
    print("Loading COMPAS Dataset")
    print("="*80)
    train_data, test_data = load_compas_data()
    print(f"✓ Train: {len(train_data.y)} samples")
    print(f"✓ Test:  {len(test_data.y)} samples")
    
    # Base hyperparameters
    base_params = {
        'hidden_dims': [128, 64],  # Adjust based on your tuning
        'learning_rate': args.learning_rate,
        'n_epochs': args.n_epochs,
        'batch_size': 64,
        'dropout': 0.2,
        'ig_steps': 32,
        'ema_rate': 0.1,
        'norm': 'l2'
    }
    
    # Run experiments
    df_ig = vary_lambda_ig(
        lambda_ig_values=args.lambda_ig_values,
        lambda_fair_fixed=args.tuned_lambda_fair,
        train_data=train_data,
        test_data=test_data,
        base_params=base_params,
        device=args.device
    )
    
    df_fair = vary_lambda_fair(
        lambda_fair_values=args.lambda_fair_values,
        lambda_ig_fixed=args.tuned_lambda_ig,
        train_data=train_data,
        test_data=test_data,
        base_params=base_params,
        device=args.device
    )
    
    # Save results
    df_ig.to_csv(output_dir / 'sensitivity_vary_lambda_ig.csv', index=False)
    df_fair.to_csv(output_dir / 'sensitivity_vary_lambda_fair.csv', index=False)
    print(f"\n✅ Saved CSV results to {output_dir}")
    
    # Create plots
    fig = plot_sensitivity_analysis(
        df_ig=df_ig,
        df_fair=df_fair,
        tuned_lambda_ig=args.tuned_lambda_ig,
        tuned_lambda_fair=args.tuned_lambda_fair,
        output_path=output_dir / 'sensitivity_analysis.png'
    )
    
    # Also save as PDF for paper
    fig.savefig(output_dir / 'sensitivity_analysis.pdf', dpi=300, bbox_inches='tight')
    
    # Print summary
    print_summary(df_ig, df_fair, args.tuned_lambda_ig, args.tuned_lambda_fair)
    
    # Save summary to file
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("LAMBDA SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Tuned values: λ_ig={args.tuned_lambda_ig}, λ_fair={args.tuned_lambda_fair}\n\n")
        f.write(f"Varying λ_ig (λ_fair={args.tuned_lambda_fair}):\n")
        f.write(f"  F1:     {df_ig['f1'].min():.4f} - {df_ig['f1'].max():.4f}\n")
        f.write(f"  EO Gap: {df_ig['equalized_odds'].min():.4f} - {df_ig['equalized_odds'].max():.4f}\n")
        f.write(f"  GCf-IG: {df_ig['gcf_ig'].min():.4f} - {df_ig['gcf_ig'].max():.4f}\n\n")
        f.write(f"Varying λ_fair (λ_ig={args.tuned_lambda_ig}):\n")
        f.write(f"  F1:     {df_fair['f1'].min():.4f} - {df_fair['f1'].max():.4f}\n")
        f.write(f"  EO Gap: {df_fair['equalized_odds'].min():.4f} - {df_fair['equalized_odds'].max():.4f}\n")
        f.write(f"  GCf-IG: {df_fair['gcf_ig'].min():.4f} - {df_fair['gcf_ig'].max():.4f}\n")
    
    print("\n" + "="*80)
    print("✅ SENSITIVITY ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - sensitivity_vary_lambda_ig.csv")
    print(f"  - sensitivity_vary_lambda_fair.csv")
    print(f"  - sensitivity_analysis.png")
    print(f"  - sensitivity_analysis.pdf")
    print(f"  - summary.txt")


if __name__ == '__main__':
    main()
