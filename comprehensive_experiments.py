"""
Comprehensive experiment runner for GCf-IG research paper.

Implements all analyses from EDiff paper:
- RQ1: Main comparison with baselines
- RQ2: Multi-objective simplex analysis
- RQ3: Ablation study
- RQ4: Efficiency analysis
"""

import numpy as np
import torch
import time
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from data_utils import load_german_credit
from models import MLPClassifier
from trainer import GCfIGTrainer
from metrics import evaluate_model, print_metrics

from baselines import train_baseline_method
from ranking_evaluation import (
    create_ranking_table,
    aggregate_ranks_across_datasets,
    generate_ranking_report,
    print_ranking_table
)
from simplex_analysis import run_simplex_analysis, compare_multiple_objectives
from additional_datasets import get_dataset


class ComprehensiveExperiment:
    """
    Main experiment class that runs all analyses.
    """
    
    def __init__(
        self,
        output_dir: str = "./results",
        random_seed: int = 42
    ):
        """
        Args:
            output_dir: Directory to save all outputs
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        self.set_seeds()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Storage for results
        self.dataset_results = {}  # {dataset_name: {method_name: results}}
        self.timing_results = {}   # {dataset_name: {method_name: time}}
        
    def set_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)
    
    def get_default_config(self) -> Dict:
        """Get default training configuration."""
        config = {
            'hidden_dims': [128, 32],
            'dropout': 0.2,
            'use_batch_norm': False,
            'n_epochs': 50,
            'batch_size': 128,
            'learning_rate': 0.003,
            'lambda_ig': 1.0,
            'lambda_sp': 0.0,  # Focus on EO, not SP
            'lambda_eo': 1.0,  # Optimize for Equalized Odds
            'lambda_fairness': 1.0,
            'lambda_adv': 1.0,
            'device': self.device,
            'seed': self.random_seed
        }
    
        # Override with tuned params if available
        if hasattr(self, 'tuned_params'):
            # Use tuned params from most recent tuning run
            print("📌 Using tuned hyperparameters")
            # Note: We'll apply method-specific params in run_single_method
    
        return config
    
    def _load_dataset(self, dataset_name: str):
        """Centralized dataset loading to keep paths consistent."""
        if dataset_name == 'german':
            # Prefer caller-provided path in load_german_credit if your function supports defaults.
            # If your loader REQUIRES a path, replace None with your local path, e.g. "german.csv".
            try:
                train_data, test_data = load_german_credit()
            except TypeError:
                # Backward-compatible fallback
                train_data, test_data = load_german_credit('german.csv')
            return train_data, test_data
        
        # adult / compas / bank should be handled by additional_datasets.get_dataset
        return get_dataset(dataset_name)

    def run_single_method(self, method_name, train_data, test_data, config, verbose=True):
        """
        Train and evaluate a single method.
        """
        # Check if we have tuned params for this method
        if hasattr(self, 'tuned_params') and method_name in self.tuned_params:
            tuned = self.tuned_params[method_name]
            print(f"  Using tuned params for {method_name}: lr={tuned.get('learning_rate', config['learning_rate']):.4f}")
        
            # Override config with tuned params
            config = {**config, **tuned}
    
    # Rest of method continues as before...
        start_time = time.time()
        
        if method_name == 'gcf_ig':
            # GCf-IG (our method)
            model = MLPClassifier(
                input_dim=train_data.X.shape[1],
                hidden_dims=config['hidden_dims'],
                dropout=config['dropout']
            )
            
            trainer = GCfIGTrainer(model, config, device=self.device)
            trainer.train(train_data, verbose=verbose)
            
            # Get baselines for evaluation
            baselines = {
                k: v.cpu().numpy()
                for k, v in trainer.baseline_manager.get_baselines().items()
            }
            
            # Evaluate
            train_metrics = evaluate_model(trainer, train_data, baselines, self.device, compute_gcf_ig=True)
            test_metrics = evaluate_model(trainer, test_data, baselines, self.device, compute_gcf_ig=True)
            model_obj = trainer
        
        else:
            # Baseline methods (includes: unconstrained, dir, hardt, agarwal, lagrangian, adversarial)
            method, method_results = train_baseline_method(
                method_name, train_data, test_data, config, self.device, verbose
            )
            train_metrics = method_results['train']
            test_metrics = method_results['test']
            model_obj = method
        
        elapsed_time = time.time() - start_time
        
        results = {
            'train': train_metrics,
            'test': test_metrics,
            'time': elapsed_time,
            'model_obj': model_obj
        }
        if method_name == 'gcf_ig':
            results['baselines'] = baselines

        return results
    def save_rq1_metrics_csv(self, filename: str = "rq1_all_metrics.csv"):
        """
        Saves one flat CSV with rows = (dataset, method) and columns = core metrics.
        """
        import pandas as pd

        rows = []
        metrics_cols = [
            "auc", "f1", "precision", "recall",
            "statistical_parity", "equalized_odds", "gcf_ig",
            "accuracy"
        ]

        for dataset_name, methods_dict in self.dataset_results.items():
            # Skip non-RQ1 keys if you store ablation there
            if dataset_name.endswith("_ablation"):
                continue

            for method_name, result in methods_dict.items():
                test_metrics = result.get("test", {})
                row = {
                    "dataset": dataset_name,
                    "method": method_name,
                    "train_time": result.get("time", np.nan)
                }
                for c in metrics_cols:
                    row[c] = test_metrics.get(c, np.nan)
                rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False)
        print(f"\nSaved RQ1 metrics table to: {out_path}")
    
    def run_rq1_main_comparison(
        self,
        datasets: List[str] = None,
        methods: List[str] = None,
        verbose: bool = True
    ):
        """
        RQ1: Compare GCf-IG with baseline methods.
        
        Generates ranking tables like Table 3 in EDiff paper.
        """
        #  Patch: default to ALL FOUR datasets
        if datasets is None:
            datasets = ['german', 'compas', 'adult', 'bank']

        #  Patch: include pre-processing DIR baseline ('dir')
        # Keep your original baselines too.
        if methods is None:
            methods = [
                'unconstrained',
                'dir',          # NEW: Feldman Disparate Impact Remover (pre-processing)
                'hardt',        # post-processing
                'agarwal',      # reductions (optional to keep)
                'lagrangian',   # in-processing
                'adversarial',  # in-processing
                'gcf_ig'        # our method
            ]
        
        print("\n" + "="*120)
        print("RQ1: MAIN COMPARISON WITH BASELINES")
        print("="*120)
        
        config = self.get_default_config()
        
        for dataset_name in datasets:
            print(f"\n{'='*120}")
            print(f"Dataset: {dataset_name.upper()}")
            print(f"{'='*120}")
            
            # ✅ Patch: use centralized loader (handles german path robustly)
            train_data, test_data = self._load_dataset(dataset_name)
            
            dataset_results = {}
            dataset_timing = {}
            
            for method_name in methods:
                print(f"\nRunning: {method_name}...")
                
                try:
                    results = self.run_single_method(
                        method_name, train_data, test_data, config, verbose=False
                    )
                    
                    dataset_results[method_name] = results
                    dataset_timing[method_name] = results['time']
                    
                    print(f"  Time: {results['time']:.2f}s")
                    print(f"  Test Accuracy: {results['test']['accuracy']:.4f}")
                    print(f"  Test EO: {results['test']['equalized_odds']:.4f}")
                    if 'gcf_ig' in results['test']:
                        print(f"  Test GCf-IG: {results['test']['gcf_ig']:.4f}")
                
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Store results
            self.dataset_results[dataset_name] = dataset_results
            self.timing_results[dataset_name] = dataset_timing
            
            # Generate ranking table for this dataset
            rank_df = create_ranking_table(dataset_results, dataset_name=dataset_name)
            print_ranking_table(rank_df, title=f"{dataset_name.upper()} - Method Rankings")
            
            # Save
            rank_df.to_csv(self.output_dir / f"rankings_{dataset_name}.csv")
        
        # Aggregated rankings across datasets
        if len(datasets) > 1:
            print("\n" + "="*120)
            print("AGGREGATED RANKINGS ACROSS ALL DATASETS")
            print("="*120)
            
            agg_df = aggregate_ranks_across_datasets(self.dataset_results)
            print("\n" + agg_df.to_string())
            agg_df.to_csv(self.output_dir / "rankings_aggregated.csv")
        self.save_rq1_metrics_csv()

    # K-Fold Cross-Validation Evaluation Framework
    # Replaces seed-based evaluation with proper 5-fold CV.

    def run_rq1_with_kfold_cv(
        self,
        datasets: List[str] = None,
        methods: List[str] = None,
        n_folds: int = 5,
        random_state: int = 42,
        quick_mode: bool = False
    ):
        """
        RQ1: Main comparison using K-Fold Cross-Validation.

        Replaces seed-based evaluation with stratified k-fold CV.

        Args:
            datasets: List of dataset names
            methods: List of method names
            n_folds: Number of folds (default: 5)
            random_state: Random seed for fold splitting

        Outputs:
            - rq1_cv_metrics_by_fold.csv: Raw metrics for each fold
            - rq1_cv_summary.csv: Mean ± std across folds
            - rq1_cv_confidence_intervals.csv: 95% CI for each metric
        """
        from sklearn.model_selection import StratifiedKFold
        import pandas as pd
        from metrics import evaluate_model, build_label_group_baselines

        if datasets is None:
            datasets = ['german', 'compas', 'adult', 'bank']
        if methods is None:
            methods = ['unconstrained', 'dir', 'hardt', 'agarwal', 'lagrangian', 'adversarial', 'gcf_ig']

        print("\n" + "="*120)
        print("RQ1: K-FOLD CROSS-VALIDATION EVALUATION")
        print("="*120)
        print(f"Datasets: {datasets}")
        print(f"Methods: {methods}")
        print(f"Folds: {n_folds}")
        print("="*120)

        all_results = []  # Store all fold results

        for dataset_name in datasets:
            print(f"\n{'='*120}")
            print(f"Dataset: {dataset_name.upper()}")
            print(f"{'='*120}")

            # Load full dataset (we'll split it ourselves with k-fold)
            full_train_data, _ = self._load_dataset(dataset_name)

            # Combine train and test for k-fold splitting
            # (We'll create our own train/test splits via CV)
            X_full = full_train_data.X
            y_full = full_train_data.y
            a_full = full_train_data.a
            feature_names = full_train_data.feature_names

            # Stratified K-Fold (stratify by both label and group)
            stratify_key = y_full * 2 + a_full
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

            for method_name in methods:
                print(f"\n{'-'*80}")
                print(f"Method: {method_name}")
                print(f"{'-'*80}")

                fold_results = []

                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, stratify_key)):
                    print(f"  Fold {fold_idx + 1}/{n_folds}...", end=" ", flush=True)

                    # Create train/test split for this fold
                    from data_utils import FairnessDataset

                    train_data = FairnessDataset(
                        X=X_full[train_idx],
                        y=y_full[train_idx],
                        a=a_full[train_idx],
                        feature_names=feature_names,
                        protected_attr_name=full_train_data.protected_attr_name,
                        group_names=full_train_data.group_names
                    )

                    test_data = FairnessDataset(
                        X=X_full[test_idx],
                        y=y_full[test_idx],
                        a=a_full[test_idx],
                        feature_names=feature_names,
                        protected_attr_name=full_train_data.protected_attr_name,
                        group_names=full_train_data.group_names
                    )

                    # Train and evaluate
                    config = self.get_default_config()
                    config['seed'] = random_state + fold_idx  # Different seed per fold

                    try:
                        results = self.run_single_method(
                            method_name, train_data, test_data, config, verbose=False
                        )

                        # Extract metrics
                        test_metrics = results['test']

                        fold_result = {
                            'dataset': dataset_name,
                            'method': method_name,
                            'fold': fold_idx + 1,
                            'train_size': len(train_idx),
                            'test_size': len(test_idx),
                            'train_time': results.get('time', np.nan),

                            # Performance metrics
                            'accuracy': test_metrics.get('accuracy', np.nan),
                            'auc': test_metrics.get('auc', np.nan),
                            'f1': test_metrics.get('f1', np.nan),
                            'precision': test_metrics.get('precision', np.nan),
                            'recall': test_metrics.get('recall', np.nan),

                            # Fairness metrics
                            'statistical_parity': test_metrics.get('statistical_parity', np.nan),
                            'equalized_odds': test_metrics.get('equalized_odds', np.nan),
                            'tpr_group0': test_metrics.get('tpr_group0', np.nan),
                            'tpr_group1': test_metrics.get('tpr_group1', np.nan),
                            'fpr_group0': test_metrics.get('fpr_group0', np.nan),
                            'fpr_group1': test_metrics.get('fpr_group1', np.nan),
                            'tpr_diff': test_metrics.get('tpr_diff', np.nan),
                            'fpr_diff': test_metrics.get('fpr_diff', np.nan),

                            # Procedural fairness
                            'gcf_ig': test_metrics.get('gcf_ig', np.nan),
                        }

                        fold_results.append(fold_result)
                        all_results.append(fold_result)

                        print(f"✓ (Acc={test_metrics.get('accuracy', 0):.3f}, GCf-IG={test_metrics.get('gcf_ig', 0):.3f})")

                    except Exception as e:
                        print(f"✗ Error: {e}")
                        import traceback
                        traceback.print_exc()

                # Print summary for this method
                if fold_results:
                    df_method = pd.DataFrame(fold_results)
                    print(f"\n  Summary across {n_folds} folds:")
                    print(f"    Accuracy:  {df_method['accuracy'].mean():.4f} ± {df_method['accuracy'].std():.4f}")
                    print(f"    EO Gap:    {df_method['equalized_odds'].mean():.4f} ± {df_method['equalized_odds'].std():.4f}")
                    print(f"    GCf-IG:    {df_method['gcf_ig'].mean():.4f} ± {df_method['gcf_ig'].std():.4f}")

        # Save raw fold results
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(self.output_dir / "rq1_cv_metrics_by_fold.csv", index=False)
        print(f"\n✓ Saved: rq1_cv_metrics_by_fold.csv")

        # Compute summary statistics (mean ± std)
        metric_cols = [
            'accuracy', 'auc', 'f1', 'precision', 'recall',
            'statistical_parity', 'equalized_odds', 'gcf_ig',
            'tpr_diff', 'fpr_diff', 'train_time'
        ]

        summary_rows = []
        for dataset_name in datasets:
            for method_name in methods:
                df_subset = df_all[(df_all['dataset'] == dataset_name) & (df_all['method'] == method_name)]

                if len(df_subset) == 0:
                    continue

                row = {
                    'dataset': dataset_name,
                    'method': method_name,
                    'n_folds': len(df_subset)
                }

                for metric in metric_cols:
                    values = df_subset[metric].dropna()
                    if len(values) > 0:
                        row[f'{metric}_mean'] = values.mean()
                        row[f'{metric}_std'] = values.std()
                        row[f'{metric}_min'] = values.min()
                        row[f'{metric}_max'] = values.max()
                    else:
                        row[f'{metric}_mean'] = np.nan
                        row[f'{metric}_std'] = np.nan
                        row[f'{metric}_min'] = np.nan
                        row[f'{metric}_max'] = np.nan

                summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(self.output_dir / "rq1_cv_summary.csv", index=False)
        print(f"✓ Saved: rq1_cv_summary.csv")

        # Compute 95% confidence intervals
        ci_rows = []
        for dataset_name in datasets:
            for method_name in methods:
                df_subset = df_all[(df_all['dataset'] == dataset_name) & (df_all['method'] == method_name)]

                if len(df_subset) == 0:
                    continue

                for metric in metric_cols:
                    values = df_subset[metric].dropna().values
                    if len(values) > 1:
                        # 95% CI using t-distribution
                        from scipy import stats as scipy_stats
                        mean = values.mean()
                        sem = scipy_stats.sem(values)  # Standard error of mean
                        ci = scipy_stats.t.interval(
                            0.95,
                            len(values) - 1,
                            loc=mean,
                            scale=sem
                        )

                        ci_rows.append({
                            'dataset': dataset_name,
                            'method': method_name,
                            'metric': metric,
                            'mean': mean,
                            'std': values.std(),
                            'ci_lower': ci[0],
                            'ci_upper': ci[1],
                            'n_folds': len(values)
                        })

        df_ci = pd.DataFrame(ci_rows)
        df_ci.to_csv(self.output_dir / "rq1_cv_confidence_intervals.csv", index=False)
        print(f"✓ Saved: rq1_cv_confidence_intervals.csv")

        print("\n" + "="*120)
        print("K-FOLD CROSS-VALIDATION COMPLETE")
        print("="*120)

        return df_all, df_summary, df_ci

    def run_rq1_with_seeds_and_ci(
        self,
        datasets=None,
        methods=None,
        seeds=None,
        n_boot=500,
        ci_alpha=0.05
    ):
        """
        RQ1 replication: run all (dataset, method) across multiple seeds, save:
          (1) per-seed raw metrics for statistical tests
          (2) bootstrap CIs (over test samples) for each seed & method

            Notes:
            - Uses y_prob/y_pred saved by evaluate_model via keys: _y_prob, _y_pred, _y_true, _a.
            - For GCf-IG, we recommend CI across seeds in the first paper (bootstrap is expensive).
        """
        import pandas as pd
        from metrics import bootstrap_metrics, build_label_group_baselines, compute_gcf_ig_metric

        if datasets is None:
            datasets = ['german', 'compas', 'adult', 'bank']
        if methods is None:
            methods = ['unconstrained', 'dir', 'lagrangian', 'adversarial', 'hardt', 'gcf_ig']
        if seeds is None:
            seeds = [0, 1, 2, 3, 4]  # change as needed

        rows = []         # per-seed raw metrics
        ci_rows = []      # per-seed bootstrap CI rows (long format)

        for dataset_name in datasets:
            train_data, test_data = self._load_dataset(dataset_name)

            for seed in seeds:
                # Set seed for reproducibility (training randomness)
                self.random_seed = seed
                self.set_seeds()

                for method in methods:
                    config = self.get_default_config()
                    config['seed'] = seed
                    config['device'] = self.device

                    results = self.run_single_method(method, train_data, test_data, config, verbose=False)
                    test_m = results['test']

                    # Get test probabilities for bootstrap
                    # You need a function to obtain y_prob from the trained method/trainer.
                    # If your evaluate_model already computed probs internally, add a helper to return them.
                    y_prob = test_m.get("_y_prob", None)
                    if y_prob is None:
                        # Minimal approach: add a metrics.get_predictions_proba(...) helper and call it here.
                        from metrics import get_predictions_proba
                        y_prob = get_predictions_proba(results.get("model_obj", None), test_data.X, device=self.device)

                    # Compute GCf-IG metric for this run if not present
                    if 'gcf_ig' not in test_m:
                        baselines = build_label_group_baselines(train_data.X, train_data.y, train_data.a)
                        gcfv, _ = compute_gcf_ig_metric(results.get("model_obj", None), test_data.X, test_data.y, test_data.a, baselines, device=self.device)
                        test_m['gcf_ig'] = gcfv

                    # Save per-seed row
                    row = {
                        "dataset": dataset_name,
                        "method": method,
                        "seed": seed,
                        "train_time": results.get("time", np.nan),

                        "accuracy": test_m.get("accuracy", np.nan),
                        "auc": test_m.get("auc", np.nan),
                        "f1": test_m.get("f1", np.nan),
                        "precision": test_m.get("precision", np.nan),
                        "recall": test_m.get("recall", np.nan),

                        "statistical_parity": test_m.get("statistical_parity", np.nan),
                        "equalized_odds": test_m.get("equalized_odds", np.nan),
                        "tpr_g0": test_m.get("tpr_group0", np.nan),
                        "tpr_g1": test_m.get("tpr_group1", np.nan),
                        "fpr_g0": test_m.get("fpr_group0", np.nan),
                        "fpr_g1": test_m.get("fpr_group1", np.nan),
                        "dtpr": test_m.get("tpr_diff", np.nan),
                        "dfpr": test_m.get("fpr_diff", np.nan),

                        "gcf_ig": test_m.get("gcf_ig", np.nan),
                    }
                    rows.append(row)

                    # Bootstrap CI on test set (excluding gcf_ig unless you want to bootstrap it too)
                    ci = bootstrap_metrics(test_data.y, y_prob, test_data.a, n_boot=n_boot, seed=seed)
                    for metric, (mean, lo, hi) in ci.items():
                        ci_rows.append({
                            "dataset": dataset_name,
                            "method": method,
                            "seed": seed,
                            "metric": metric,
                            "mean": mean,
                            "ci_low": lo,
                            "ci_high": hi
                        })

        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "rq1_metrics_by_seed.csv", index=False)

        df_ci = pd.DataFrame(ci_rows)
        df_ci.to_csv(self.output_dir / "rq1_bootstrap_ci_by_seed.csv", index=False)

        # Aggregate across seeds for paper table
        summary = (df
            .groupby(["dataset", "method"])
            .agg(["mean", "std"])
            .reset_index()
        )
        summary.to_csv(self.output_dir / "rq1_summary_across_seeds.csv", index=False)

        print("Saved:")
        print(" - rq1_metrics_by_seed.csv")
        print(" - rq1_bootstrap_ci_by_seed.csv")
        print(" - rq1_summary_across_seeds.csv")



    def run_rq2_simplex_analysis(
        self,
        dataset_name: str = 'adult',
        objective_sets: List[tuple] = None
    ):
        """
        RQ2: Multi-objective simplex analysis.
        
        Generates simplex plots like Figure 4 and coverage table like Table 4.
        """
        print("\n" + "="*120)
        print("RQ2: MULTI-OBJECTIVE SIMPLEX ANALYSIS")
        print("="*120)
        
        if dataset_name not in self.dataset_results:
            print(f"No results for {dataset_name}. Run RQ1 first.")
            return
        
        results = self.dataset_results[dataset_name]
        
        # Default objective sets
        if objective_sets is None:
            objective_sets = [
                ('auc', 'statistical_parity', 'gcf_ig'),
                ('auc', 'equalized_odds', 'gcf_ig'),
                ('f1', 'statistical_parity', 'gcf_ig'),
            ]
        
        # Run simplex analysis for each objective set
        simplex_dir = self.output_dir / "simplex"
        simplex_dir.mkdir(exist_ok=True)
        
        all_analyses = compare_multiple_objectives(
            results,
            objective_sets,
            save_dir=str(simplex_dir)
        )
        
        # Save coverage data
        coverage_summary = {}
        for obj_set, analysis in all_analyses.items():
            coverage_summary[str(obj_set)] = analysis['coverage']
        
        with open(simplex_dir / "coverage_summary.json", 'w') as f:
            json.dump(coverage_summary, f, indent=2)
    
    def run_rq3_ablation_study(
        self,
        dataset_name: str = 'german'
    ):
        """
        RQ3: Ablation study.
        
        Tests different loss combinations like Table 5 in EDiff paper.
        """
        print("\n" + "="*120)
        print("RQ3: ABLATION STUDY")
        print("="*120)
        
        # Patch: use centralized loader
        train_data, test_data = self._load_dataset(dataset_name)
        
        # Configurations to test
        configurations = {
            'Pred only': {
                'lambda_ig': 0.0,
                'lambda_eo': 0.0,
                'n_epochs': 30,
            },
            'Pred + SP': {
                'lambda_ig': 0.0,
                'lambda_eo': 1.0,
                'n_epochs': 30,
            },
            'Pred + GCf-IG': {
                'lambda_ig': 1.0,
                'lambda_eo': 0.0,
                'n_epochs': 30,
            },
            'Full (Pred + SP + GCf-IG)': {
                'lambda_ig': 1.0,
                'lambda_eo': 1.0,
                'n_epochs': 30,
            },
        }
        
        ablation_results = {}
        
        for name, partial_config in configurations.items():
            print(f"\n{'='*80}")
            print(f"Running: {name}")
            print(f"{'='*80}")
            
            # Use default config and update
            config = self.get_default_config()
            config.update(partial_config)
            
            try:
                results = self.run_single_method(
                    'gcf_ig', train_data, test_data, config, verbose=False
                )
                ablation_results[name] = results
                
                print(f"  Test Accuracy: {results['test']['accuracy']:.4f}")
                print(f"  Test SP: {results['test']['statistical_parity']:.4f}")
                print(f"  Test EO: {results['test']['equalized_odds']:.4f}")
                if 'gcf_ig' in results['test']:
                    print(f"  Test GCf-IG: {results['test']['gcf_ig']:.4f}")
            
            except Exception as e:
                print(f"  Error: {e}")
        
        # Create summary table
        import pandas as pd
        
        metrics_to_show = ['auc', 'f1', 'statistical_parity', 'equalized_odds', 'gcf_ig']
        
        summary_data = []
        for name, result in ablation_results.items():
            row = {'Configuration': name}
            for metric in metrics_to_show:
                value = result['test'].get(metric, np.nan)
                row[metric] = value
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "="*120)
        print("ABLATION STUDY SUMMARY")
        print("="*120)
        print("\n" + summary_df.to_string(index=False))
        
        # Save
        summary_df.to_csv(self.output_dir / f"ablation_{dataset_name}.csv", index=False)
        
        # Store for later
        self.dataset_results[f"{dataset_name}_ablation"] = ablation_results
    
    def run_rq4_efficiency_analysis(self):
        """
        RQ4: Computational efficiency analysis.
        
        Creates timing table like Table 6 in EDiff paper.
        """
        print("\n" + "="*120)
        print("RQ4: COMPUTATIONAL EFFICIENCY ANALYSIS")
        print("="*120)
        
        if not self.timing_results:
            print("No timing data. Run RQ1 first.")
            return
        
        import pandas as pd
        
        # Create timing table
        timing_data = []
        for dataset_name, method_times in self.timing_results.items():
            for method_name, time_sec in method_times.items():
                timing_data.append({
                    'Dataset': dataset_name,
                    'Method': method_name,
                    'Time (seconds)': time_sec
                })
        
        timing_df = pd.DataFrame(timing_data)
        timing_pivot = timing_df.pivot(index='Method', columns='Dataset', values='Time (seconds)')
        
        print("\n" + timing_pivot.to_string())
        
        # Save
        timing_pivot.to_csv(self.output_dir / "efficiency_analysis.csv")
        
        # Compute relative slowdowns
        print("\n" + "="*80)
        print("Relative Slowdown (vs. Unconstrained)")
        print("="*80)
        
        for dataset in timing_pivot.columns:
            baseline_time = timing_pivot.loc['unconstrained', dataset]
            print(f"\n{dataset.upper()}:")
            for method in timing_pivot.index:
                method_time = timing_pivot.loc[method, dataset]
                slowdown = method_time / baseline_time
                print(f"  {method:<30} {slowdown:>6.2f}x")
    

# new rq5

    def run_rq5_statistical_significance(
        self,
        reference_method: str = 'gcf_ig',
        metrics: List[str] = None
    ):
        """
        RQ5: Statistical significance testing.
    
        Tests if GCf-IG significantly outperforms baselines.
    
        Args:
            reference_method: Method to compare against (typically 'gcf_ig')
            metrics: List of metrics to test
        """
        from ranking_evaluation import (
            generate_significance_report,
            compute_pairwise_significance_matrix,
            create_significance_heatmap
        )
    
        print("\n" + "="*120)
        print("RQ5: STATISTICAL SIGNIFICANCE ANALYSIS")
        print("="*120)
    
        if not self.dataset_results:
            print("No results available. Run RQ1 first.")
            return
    
        # Generate significance report
        generate_significance_report(
            self.dataset_results,
            reference_method=reference_method,
            metrics=metrics,
            save_dir=str(self.output_dir)
        )
    
        # Create significance matrices for key metrics
        if metrics is None:
            metrics = ['auc', 'equalized_odds', 'gcf_ig']
    
        method_names = list(list(self.dataset_results.values())[0].keys())
    
        for metric in metrics:
            print(f"\n\nComputing pairwise significance matrix for: {metric}")
        
            sig_matrix = compute_pairwise_significance_matrix(
                self.dataset_results,
                metric=metric,
                method_names=method_names
            )
        
            # Save matrix
            sig_matrix.to_csv(self.output_dir / f"significance_matrix_{metric}.csv")
        
            # Create heatmap
            create_significance_heatmap(
                sig_matrix,
                metric_name=metric,
                save_path=str(self.output_dir / f"significance_heatmap_{metric}.png")
            )
    
        print("\n" + "="*120)
        print("Statistical significance analysis complete")
        print("="*120)

    # Enhanced RQ5: Statistical Significance using CV Fold Data
    # This version uses the fold-by-fold results from RQ1 for more robust statistical tests.

    def run_rq5_statistical_significance_cv(
        self,
        reference_method: str = 'gcf_ig',
        metrics: List[str] = None
    ):
        """
        RQ5: Statistical significance testing using CV fold results.
    
        Uses fold-by-fold data from RQ1 k-fold CV to perform paired statistical tests.
        This is more rigorous than comparing single runs.
    
        Args:
        reference_method: Method to compare against (typically 'gcf_ig')
        metrics: List of metrics to test
        """
        if metrics is None:
            metrics = ['accuracy', 'f1', 'equalized_odds', 'statistical_parity', 'gcf_ig']
    
        print("\n" + "="*120)
        print("RQ5: STATISTICAL SIGNIFICANCE ANALYSIS (using CV folds)")
        print("="*120)
    
        if not hasattr(self, 'cv_fold_results'):
            print(" No CV fold results available. Run RQ1 with k-fold CV first.")
            return
    
        df_folds = self.cv_fold_results
        datasets = df_folds['dataset'].unique()
    
        # Paired t-test results
        test_results = []
    
        for dataset_name in datasets:
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name.upper()}")
            print(f"{'='*80}")
        
            df_ds = df_folds[df_folds['dataset'] == dataset_name]
            methods = df_ds['method'].unique()
            other_methods = [m for m in methods if m != reference_method]
        
            for metric in metrics:
                print(f"\n  Metric: {metric}")
                print(f"  {'-'*60}")
            
                # Get reference method fold scores
                ref_scores = df_ds[df_ds['method'] == reference_method][metric].values
            
                if len(ref_scores) == 0:
                    print(f"   {reference_method} not found in results")
                    continue
            
                for other_method in other_methods:
                    other_scores = df_ds[df_ds['method'] == other_method][metric].values
                
                    if len(other_scores) == 0 or len(ref_scores) != len(other_scores):
                        continue
                
                    # Paired t-test (since same folds)
                    from scipy.stats import wilcoxon
                    try:
                        t_stat, p_value = wilcoxon(ref_scores, other_scores)
                    except ValueError:
                        t_stat, p_value = 0.0, 1.0
                    
                    # Effect size (Cohen's d for paired samples)
                    diff = ref_scores - other_scores
                    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
                
                    # Determine significance
                    if p_value < 0.001:
                        sig = "***"
                    elif p_value < 0.01:
                        sig = "**"
                    elif p_value < 0.05:
                        sig = "*"
                    else:
                        sig = "n.s."
                
                    # Determine direction (better/worse)
                    ref_mean = ref_scores.mean()
                    other_mean = other_scores.mean()
                
                    # For fairness metrics (lower is better)
                    if metric in ['equalized_odds', 'statistical_parity', 'gcf_ig']:
                        if ref_mean < other_mean:
                            direction = "BETTER"
                        else:
                            direction = "WORSE"
                    else:
                        # For performance metrics (higher is better)
                        if ref_mean > other_mean:
                            direction = "BETTER"
                        else:
                            direction = "WORSE"
                
                    print(f"    {reference_method} vs {other_method:<15} | "
                        f"t={t_stat:6.3f}, p={p_value:.4f} {sig:>4} | "
                        f"d={cohens_d:+.3f} | {direction}")
                
                    # Store result
                    test_results.append({
                        'dataset': dataset_name,
                        'metric': metric,
                        'reference_method': reference_method,
                        'comparison_method': other_method,
                        'reference_mean': ref_mean,
                        'comparison_mean': other_mean,
                        'w_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': sig,
                        'direction': direction,
                        'n_folds': len(ref_scores)
                    })
    
        # Save results
        df_tests = pd.DataFrame(test_results)
        df_tests.to_csv(self.output_dir / "rq5_statistical_tests_cv.csv", index=False)
        print(f"\n✓ Saved: rq5_statistical_tests_cv.csv")
    
        # Summary table
        print("\n" + "="*120)
        print("STATISTICAL SIGNIFICANCE SUMMARY")
        print("="*120)
        print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
        print("Cohen's d: |d|<0.2 (small), 0.2-0.5 (medium), >0.5 (large)")
        print("\nKey findings:")
    
        # Count significant results
        sig_df = df_tests[df_tests['p_value'] < 0.05]
        print(f"  - {len(sig_df)} / {len(df_tests)} comparisons are statistically significant (p<0.05)")
    
        # GCf-IG metric specifically
        gcfig_tests = df_tests[df_tests['metric'] == 'gcf_ig']
        gcfig_better = gcfig_tests[
            (gcfig_tests['direction'] == 'BETTER') & 
            (gcfig_tests['p_value'] < 0.05)
        ]
        print(f"  - {reference_method} has significantly better GCf-IG in {len(gcfig_better)} / {len(gcfig_tests)} comparisons")
    
        # Performance trade-offs
        acc_tests = df_tests[df_tests['metric'] == 'accuracy']
        acc_worse = acc_tests[
            (acc_tests['direction'] == 'WORSE') & 
            (acc_tests['p_value'] < 0.05)
        ]
        if len(acc_worse) > 0:
            print(f"  - {reference_method} has significantly worse accuracy in {len(acc_worse)} / {len(acc_tests)} comparisons")
    
        print("\n" + "="*120)

        return df_tests

    # RQ6: Feature Importance Analysis (Figure 7 equivalent)

    def run_rq6_feature_importance_analysis(
        self,
        dataset_name: str = 'adult',
        best_method: str = 'gcf_ig',
        worst_method: str = 'unconstrained',
        top_k: int = 10
    ):
        """
        RQ6: Feature importance visualization (Figure 7 from EDiff).
    
        Compares feature attributions between best and worst performing models.
    
        Args:
            dataset_name: Dataset to analyze
            best_method: Method with best GCf-IG score
            worst_method: Method with worst GCf-IG score
            top_k: Number of top features to show
        """
        from ranking_evaluation import (
            plot_feature_importance_comparison,
            create_disparity_barplot
        )
        from metrics import build_label_group_baselines
    
        print("\n" + "="*120)
        print("RQ6: FEATURE IMPORTANCE ANALYSIS")
        print("="*120)
    
        # Load dataset
        train_data, test_data = self._load_dataset(dataset_name)
    
        # Build baselines
        baselines = build_label_group_baselines(
            train_data.X, train_data.y, train_data.a, method="mean"
        )
    
        # Check if we have stored model objects
        if dataset_name not in self.dataset_results:
            print(f"Dataset {dataset_name} not in results. Running experiments...")
            config = self.get_default_config()
        
            # Train both methods
            best_results = self.run_single_method(
                best_method, train_data, test_data, config, verbose=False
            )
            worst_results = self.run_single_method(
                worst_method, train_data, test_data, config, verbose=False
            )
        
            best_model = best_results.get('model_obj')
            worst_model = worst_results.get('model_obj')
        else:
            # Get from stored results
            best_model = self.dataset_results[dataset_name][best_method].get('model_obj')
            worst_model = self.dataset_results[dataset_name][worst_method].get('model_obj')
    
        if best_model is None or worst_model is None:
            print("Error: Model objects not available. Cannot compute feature importance.")
            return
    
        # Get feature names
        feature_names = train_data.feature_names if hasattr(train_data, 'feature_names') else \
                   [f"Feature {i}" for i in range(train_data.X.shape[1])]
    
        # Compare feature importance
        print(f"\nComparing {best_method} (best) vs. {worst_method} (worst)...")
    
        plot_feature_importance_comparison(
            best_model,
            worst_model,
            test_data,
            baselines,
            feature_names=feature_names,
            model1_name=f"{best_method} (GCf-IG={self.dataset_results.get(dataset_name, {}).get(best_method, {}).get('test', {}).get('gcf_ig', 'N/A'):.3f})" if dataset_name in self.dataset_results else best_method,
            model2_name=f"{worst_method} (GCf-IG={self.dataset_results.get(dataset_name, {}).get(worst_method, {}).get('test', {}).get('gcf_ig', 'N/A'):.3f})" if dataset_name in self.dataset_results else worst_method,
            top_k=top_k,
            save_path=str(self.output_dir / f"feature_importance_comparison_{dataset_name}.png")
        )
    
        # Individual disparity plots
        print(f"\nCreating disparity plot for {best_method}...")
        create_disparity_barplot(
            best_model,
            test_data,
            baselines,
            feature_names=feature_names,
            model_name=best_method,
            top_k=top_k,
            save_path=str(self.output_dir / f"feature_disparity_{best_method}_{dataset_name}.png")
        )
    
        print(f"\nCreating disparity plot for {worst_method}...")
        create_disparity_barplot(
            worst_model,
            test_data,
            baselines,
            feature_names=feature_names,
            model_name=worst_method,
            top_k=top_k,
            save_path=str(self.output_dir / f"feature_disparity_{worst_method}_{dataset_name}.png")
        )
    
        print("\n" + "="*120)
        print("Feature importance analysis complete")
        print("="*120)

    # Comprehensive Hyperparameter Tuning for All Methods
    # Tunes hyperparameters for ALL neural network baselines to ensure fair comparison.
    # Strategy:
    # 1. Tune base architecture (lr, hidden_dims, dropout) using Unconstrained MLP
    # 2. Use same base architecture for all methods
    # 3. Tune method-specific fairness parameters (λ_fairness, λ_adv, λ_ig, etc.)

    def tune_all_hyperparameters(
        self,
        dataset_name: str = 'adult',
        methods: List[str] = None,
        validation_split: float = 0.2,
        n_trials_base: int = 15,
        n_trials_method: int = 20,
        random_state: int = 42
    ) -> Dict[str, Dict]:
        """
        Tune hyperparameters for all neural network methods.
    
        Two-stage tuning:
        1. Find best base architecture (lr, hidden_dims, dropout) using Unconstrained
        2. For each method, tune fairness-specific parameters with fixed base architecture
    
        Args:
            dataset_name: Dataset to tune on
            methods: List of methods to tune (if None, uses all neural methods)
            validation_split: Fraction of data for validation
            n_trials_base: Number of trials for base architecture tuning
            n_trials_method: Number of trials for method-specific tuning
            random_state: Random seed
        
        Returns:
            all_best_params: Dict mapping method_name -> best_params
        """
        if methods is None:
            # Only neural network methods need tuning
            methods = ['unconstrained', 'lagrangian', 'adversarial', 'gcf_ig']
    
        print("\n" + "="*120)
        print("COMPREHENSIVE HYPERPARAMETER TUNING FOR ALL METHODS")
        print("="*120)
        print(f"Dataset: {dataset_name}")
        print(f"Methods: {methods}")
        print(f"Validation split: {validation_split:.0%}")
        print(f"Base architecture trials: {n_trials_base}")
        print(f"Method-specific trials: {n_trials_method}")
        print("="*120)
    
        # Load and split data
        from sklearn.model_selection import train_test_split
        from data_utils import FairnessDataset
    
        full_train, full_test = self._load_dataset(dataset_name)
    
        stratify_key = full_train.y * 2 + full_train.a
        X_train, X_val, y_train, y_val, a_train, a_val = train_test_split(
            full_train.X, full_train.y, full_train.a,
            test_size=validation_split,
            random_state=random_state,
            stratify=stratify_key
        )
    
        train_data = FairnessDataset(
            X=X_train, y=y_train, a=a_train,
            feature_names=full_train.feature_names,
            protected_attr_name=full_train.protected_attr_name,
            group_names=full_train.group_names
        )
    
        val_data = FairnessDataset(
            X=X_val, y=y_val, a=a_val,
            feature_names=full_train.feature_names,
            protected_attr_name=full_train.protected_attr_name,
            group_names=full_train.group_names
        )
    
        print(f"\nData split: Train={len(train_data.X)}, Val={len(val_data.X)}, Test={len(full_test.X)}")
    
        # =========================================================================
        # STAGE 1: Tune Base Architecture using Unconstrained MLP
        # =========================================================================
        print("\n" + "="*120)
        print("STAGE 1: TUNING BASE ARCHITECTURE (using Unconstrained MLP)")
        print("="*120)
    
        base_param_grid = {
            'learning_rate': [0.0001, 0.0003, 0.001, 0.003, 0.01],
            'hidden_dims': [[64, 32], [128, 64], [64, 64], [128, 32]],
            'dropout': [0.0, 0.1, 0.2],
            'batch_size': [32, 64, 128],
            'n_epochs': [50]
        }
    
        best_base_params = self._tune_single_method(
            method_name='unconstrained',
            train_data=train_data,
            val_data=val_data,
            param_grid=base_param_grid,
            n_trials=n_trials_base,
            random_state=random_state,
            optimization_metric='f1'  # Use F1 for base architecture
        )
    
        print(f"\n✅ Best base architecture:")
        print(f"   lr={best_base_params['learning_rate']:.4f}")
        print(f"   hidden_dims={best_base_params['hidden_dims']}")
        print(f"   dropout={best_base_params['dropout']:.2f}")
        print(f"   batch_size={best_base_params['batch_size']}")
        
        # CRITICAL: Convert numpy types to Python natives IMMEDIATELY
        import numpy as np
        if isinstance(best_base_params['learning_rate'], np.floating):
            best_base_params['learning_rate'] = float(best_base_params['learning_rate'])
        if isinstance(best_base_params['dropout'], np.floating):
            best_base_params['dropout'] = float(best_base_params['dropout'])
        if isinstance(best_base_params['batch_size'], np.integer):
            best_base_params['batch_size'] = int(best_base_params['batch_size'])
        if isinstance(best_base_params['hidden_dims'], list):
            best_base_params['hidden_dims'] = [int(x) if isinstance(x, np.integer) else x 
                                                for x in best_base_params['hidden_dims']]
        
        print(f"✓ Converted to Python types: batch_size type = {type(best_base_params['batch_size'])}")
    
        # =========================================================================
        # STAGE 2: Tune Method-Specific Parameters
        # =========================================================================
        print("\n" + "="*120)
        print("STAGE 2: TUNING METHOD-SPECIFIC PARAMETERS (fixed base architecture)")
        print("="*120)
    
        all_best_params = {}
    
        for method_name in methods:
            print(f"\n{'-'*120}")
            print(f"Tuning: {method_name.upper()}")
            print(f"{'-'*120}")
        
            if method_name == 'unconstrained':
                # Already tuned in Stage 1
                all_best_params[method_name] = best_base_params
                print("✅ Using base architecture params (already tuned)")
            
            elif method_name == 'lagrangian':
                # Tune λ_fairness only
                method_param_grid = {
                    'lambda_fairness': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    # Fixed base architecture
                    'learning_rate': [best_base_params['learning_rate']],
                    'hidden_dims': [best_base_params['hidden_dims']],
                    'dropout': [best_base_params['dropout']],
                    'batch_size': [best_base_params['batch_size']],
                    'n_epochs': [50]
                }
            
                best_params = self._tune_single_method(
                    method_name='lagrangian',
                    train_data=train_data,
                    val_data=val_data,
                    param_grid=method_param_grid,
                    n_trials=min(n_trials_method, 7),  # Only 7 λ values
                    random_state=random_state,
                    optimization_metric='f1'
                )
                all_best_params[method_name] = best_params
                print(f"✅ Best params: {best_params}")
            
            elif method_name == 'adversarial':
                # Tune λ_adv only
                method_param_grid = {
                    'lambda_adv': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                    # Fixed base architecture
                    'learning_rate': [best_base_params['learning_rate']],
                    'hidden_dims': [best_base_params['hidden_dims']],
                    'dropout': [best_base_params['dropout']],
                    'batch_size': [best_base_params['batch_size']],
                    'n_epochs': [50]
                }
            
                best_params = self._tune_single_method(
                    method_name='adversarial',
                    train_data=train_data,
                    val_data=val_data,
                    param_grid=method_param_grid,
                    n_trials=min(n_trials_method, 6),
                    random_state=random_state,
                    optimization_metric='f1'
                )
                all_best_params[method_name] = best_params
                print(f"✅ Best params: {best_params}")
            
            elif method_name == 'gcf_ig':
                # Tune λ_ig, λ_sp, λ_eo
                # Convert base params to Python native types (critical for DataLoader)
                import numpy as np
                method_param_grid = {
                    'lambda_ig': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'lambda_sp': [0.0],  # Always 0 - focus on EO not SP
                    'lambda_eo': [0.0, 0.5, 1.0, 2.0, 5.0],  # Main fairness constraint
                    # Fixed base architecture - USE PYTHON NATIVE TYPES
                    'learning_rate': [float(best_base_params['learning_rate']) if isinstance(best_base_params['learning_rate'], np.floating) else best_base_params['learning_rate']],
                    'hidden_dims': [[int(x) if isinstance(x, np.integer) else x for x in best_base_params['hidden_dims']]],
                    'dropout': [float(best_base_params['dropout']) if isinstance(best_base_params['dropout'], np.floating) else best_base_params['dropout']],
                    'batch_size': [int(best_base_params['batch_size']) if isinstance(best_base_params['batch_size'], np.integer) else best_base_params['batch_size']],
                    'n_epochs': [50]
                }
            
                try:
                    best_params = self._tune_single_method(
                        method_name='gcf_ig',
                        train_data=train_data,
                        val_data=val_data,
                        param_grid=method_param_grid,
                        n_trials=n_trials_method,
                        random_state=random_state,
                        optimization_metric='f1'
                    )
                    all_best_params[method_name] = best_params
                    print(f"✅ Best params: {best_params}")
                except Exception as e:
                    # Fallback: Use Stage 1 base architecture + default lambdas
                    print(f"⚠️  GCf-IG tuning failed: {str(e)[:100]}")
                    print(f"   Using Stage 1 base architecture as fallback")
                    all_best_params[method_name] = {
                        'learning_rate': float(best_base_params['learning_rate']) if isinstance(best_base_params['learning_rate'], np.floating) else best_base_params['learning_rate'],
                        'hidden_dims': [int(x) if isinstance(x, np.integer) else x for x in best_base_params['hidden_dims']],
                        'dropout': float(best_base_params['dropout']) if isinstance(best_base_params['dropout'], np.floating) else best_base_params['dropout'],
                        'batch_size': int(best_base_params['batch_size']) if isinstance(best_base_params['batch_size'], np.integer) else best_base_params['batch_size'],
                        'n_epochs': 50,
                        'lambda_ig': 1.0,
                        'lambda_sp': 0.0,
                        'lambda_eo': 1.0
                    }
                    print(f"✅ Using fallback params (Stage 1 base + default lambdas)")
        # After line 1379, before _save_tuned_params
        import numpy as np

        # Convert numpy types to Python natives
        for method, params in all_best_params.items():
            for key, val in params.items():
                if isinstance(val, np.integer):
                    params[key] = int(val)
                elif isinstance(val, np.floating):
                    params[key] = float(val)
                elif isinstance(val, list):
                    params[key] = [int(x) if isinstance(x, np.integer) else x for x in val]

        # Save all tuned parameters
        self._save_tuned_params(all_best_params, dataset_name)
    
        # Print final summary
        print("\n" + "="*120)
        print("HYPERPARAMETER TUNING COMPLETE - ALL METHODS")
        print("="*120)
        print(f"\nTuned parameters saved for {len(all_best_params)} methods:")
        for method_name in all_best_params:
            print(f"  ✅ {method_name}")
        print("\n" + "="*120)
    
        return all_best_params

    def _tune_single_method(
        self,
        method_name: str,
        train_data,
        val_data,
        param_grid: Dict,
        n_trials: int,
        random_state: int,
        optimization_metric: str = 'f1'
    ) -> Dict:
        """
        Tune hyperparameters for a single method.
    
        Internal helper method - called by tune_all_hyperparameters.
        """
        print(f"\n  Parameter grid:")
        for param, values in param_grid.items():
            if len(values) > 1:
                print(f"    {param}: {values}")
    
        # Generate random combinations
        np.random.seed(random_state)
        trials = []
    
        param_names = list(param_grid.keys())
        for _ in range(n_trials):
            params = {}
            for param_name, param_range in param_grid.items():
                # Sample from parameter range
                if isinstance(param_range[0], (int, float, str)):
                    value = np.random.choice(param_range)
                else:
                    value = param_range[np.random.randint(len(param_range))]
                
                # CRITICAL: Convert numpy types to Python natives IMMEDIATELY
                if isinstance(value, np.integer):
                    value = int(value)
                elif isinstance(value, np.floating):
                    value = float(value)
                elif isinstance(value, list):
                    value = [int(x) if isinstance(x, np.integer) else 
                            float(x) if isinstance(x, np.floating) else x 
                            for x in value]
                
                params[param_name] = value
            
            trials.append(params)
    
        # Remove duplicates
        unique_trials = []
        seen = set()
        for trial in trials:
            # Convert lists to tuples for hashing
            hashable_trial = {}
            for k, v in trial.items():
                if isinstance(v, list):
                    hashable_trial[k] = tuple(v)
                else:
                    hashable_trial[k] = v
            trial_tuple = tuple(sorted(hashable_trial.items()))
            if trial_tuple not in seen:
                seen.add(trial_tuple)
                unique_trials.append(trial)
    
        print(f"\n  Running {len(unique_trials)} unique combinations...")
    
        # Evaluate each
        results = []
        for idx, params in enumerate(unique_trials):
            try:
                config = self.get_default_config()
                config.update(params)
                
                # Train and evaluate
                method_results = self.run_single_method(
                    method_name, train_data, val_data, config, verbose=False
                )
                
                val_metrics = method_results['test']
                
                result = {
                    'trial': idx + 1,
                    **params,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_statistical_parity': val_metrics['statistical_parity'],
                    'val_equalized_odds': val_metrics['equalized_odds'],
                    'val_gcf_ig': val_metrics.get('gcf_ig', np.nan)
                }
                results.append(result)
                
                if (idx + 1) % 5 == 0:
                    print(f"    Completed {idx+1}/{len(unique_trials)} trials...")
                    
            except Exception as e:
                print(f"    Trial {idx+1} failed: {e}")
        
        # Find best
        df_results = pd.DataFrame(results)
        if len(results) == 0:
            print("\n  ❌ ERROR: All tuning trials failed!")
            print("  Returning default parameters...")
            default = self.get_default_config()
            return {
                'learning_rate': default['learning_rate'],
                'hidden_dims': default['hidden_dims'],
                'dropout': default['dropout'],
                'batch_size': default['batch_size'],
                'n_epochs': default['n_epochs']
            }


        # Check if the optimization metric exists
        if f'val_{optimization_metric}' not in df_results.columns:
            print(f"⚠️  Warning: {optimization_metric} not available, using f1 instead")
            optimization_metric = 'f1'

        if optimization_metric in ['gcf_ig', 'statistical_parity', 'equalized_odds']:
            best_idx = df_results[f'val_{optimization_metric}'].idxmin()  # Lower is better
        else:
            best_idx = df_results[f'val_{optimization_metric}'].idxmax()  # Higher is better
        
        best_result = df_results.loc[best_idx]
        best_params = {k: v for k, v in best_result.items() if k.startswith(tuple(param_grid.keys()))}
        
        print(f"\n  Best {optimization_metric}: {best_result[f'val_{optimization_metric}']:.4f}")
        
        return dict(best_params)

    def _save_tuned_params(self, all_params: Dict[str, Dict], dataset_name: str):
        """Save tuned parameters to CSV and JSON."""
        # Save as CSV (one row per method)
        rows = []
        for method_name, params in all_params.items():
            row = {'method': method_name, 'dataset': dataset_name}
            row.update(params)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / f"tuned_hyperparameters_{dataset_name}.csv", index=False)
        print(f"\n✓ Saved: tuned_hyperparameters_{dataset_name}.csv")
        
        # Also save as JSON for easy loading
        import json
        with open(self.output_dir / f"tuned_hyperparameters_{dataset_name}.json", 'w') as f:
            json.dump(all_params, f, indent=2)
        print(f"✓ Saved: tuned_hyperparameters_{dataset_name}.json")

    # Updated run_all_experiments to use K-Fold CV instead of seeds

    def run_all_experiments(
        self,
        datasets: List[str] = None,
        quick_mode: bool = False,
        n_folds: int = 5
    ):
        """
        Run all experiments (RQ1-6).
    
        RQ1 uses k-fold cross-validation for robust evaluation.
        RQ2-6 use single train/test splits.
    
        Args:
        datasets: List of datasets to use
        quick_mode: If True, use fewer epochs and methods for faster testing
        n_folds: Number of folds for cross-validation in RQ1 (default: 5)
        """
        if datasets is None:
            datasets = ['german', 'compas', 'adult', 'bank']

        if quick_mode:
            print("\n⚡ QUICK MODE: Using reduced epochs and methods for testing\n")
            self.dataset_results = {}
        
            config = self.get_default_config()
            config['n_epochs'] = 3
    
        methods = ['unconstrained', 'dir', 'hardt', 'agarwal', 'lagrangian', 'adversarial', 'gcf_ig'] if quick_mode else None
    
        # ========================================================================
        # RQ1: Main comparison with K-Fold CV (ONLY RQ1 uses CV!)
        # ========================================================================
        print("\n" + "="*120)
        print("RUNNING RQ1: MAIN COMPARISON WITH K-FOLD CROSS-VALIDATION")
        print("="*120)
    
        # Run k-fold cross-validation for RQ1
        df_folds, df_summary, df_ci = self.run_rq1_with_kfold_cv(
            datasets=datasets,
            methods=methods,
            n_folds=n_folds
        )
    
        # Store CV summary for RQ5 statistical tests
        self.cv_fold_results = df_folds
        self.cv_summary = df_summary
        self.cv_confidence_intervals = df_ci
    
        # Also populate dataset_results for compatibility with other RQs
        # This uses mean values from CV for plotting/ranking
        self.dataset_results = {}
        for dataset_name in datasets:
            self.dataset_results[dataset_name] = {}
            df_ds = df_summary[df_summary['dataset'] == dataset_name]
        
            for _, row in df_ds.iterrows():
                method = row['method']
                self.dataset_results[dataset_name][method] = {
                    'test': {
                        'accuracy': row['accuracy_mean'],
                        'auc': row['auc_mean'],
                        'f1': row['f1_mean'],
                        'statistical_parity': row['statistical_parity_mean'],
                        'equalized_odds': row['equalized_odds_mean'],
                        'gcf_ig': row['gcf_ig_mean'],
                        'tpr_diff': row['tpr_diff_mean'],
                        'fpr_diff': row['fpr_diff_mean'],
                    },
                    'time': row['train_time_mean']
             }
    
       
        # ========================================================================
        # RQ2-6: Single train/test runs (no CV needed)
        # ========================================================================
        print("\n" + "="*120)
        print("NOTE: RQ2-6 use single train/test splits (not CV)")
        print("="*120)
    
        # For RQ2-6, we need fresh single-split datasets
        # We'll train models once per dataset for these analyses
    
        # RQ2: Simplex analysis (on first dataset)
        print("\n" + "="*120)
        print("RUNNING RQ2: SIMPLEX ANALYSIS (single run)")
        print("="*120)
        self.run_rq2_simplex_analysis(datasets[0])  # Uses first dataset
    
        # RQ3: Ablation study (on first dataset)
        print("\n" + "="*120)
        print("RUNNING RQ3: ABLATION STUDY (single run)")
        print("="*120)
        self.run_rq3_ablation_study(datasets[1] if len(datasets) > 1 else datasets[0])  # Uses second dataset if available
    
        # RQ4: Efficiency analysis
        print("\n" + "="*120)
        print("RUNNING RQ4: EFFICIENCY ANALYSIS (uses RQ1 CV timing)")
        print("="*120)
        # RQ4 already uses timing from dataset_results which now has CV means
        self.run_rq4_efficiency_analysis()
    
        # RQ5: Statistical significance (uses RQ1 CV fold data!)
        if len(datasets) > 1:
            print("\n" + "="*120)
            print("RUNNING RQ5: STATISTICAL SIGNIFICANCE (uses RQ1 CV folds)")
            print("="*120)
        
            # Enhanced statistical tests using CV fold data
            self.run_rq5_statistical_significance_cv(
                reference_method='gcf_ig',
                metrics=['accuracy', 'f1', 'equalized_odds', 'statistical_parity', 'gcf_ig']
            )
    
        # RQ6: Feature importance (single run)
        print("\n" + "="*120)
        print("RUNNING RQ6: FEATURE IMPORTANCE ANALYSIS (single run)")
        print("="*120)
        self.run_rq6_feature_importance_analysis(
            dataset_name=datasets[2] if len(datasets) > 2 else datasets[0],  # Use Adult (3rd) if available
            best_method='gcf_ig',
            worst_method='unconstrained',
            top_k=10
        )
    
        # Final summary
        print("\n" + "="*120)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*120)
        print(f"\nResults saved to: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.rglob("*")):
            if file.is_file():
                print(f"  - {file.relative_to(self.output_dir)}")
    
        print("\n" + "="*120)
        print("EXPERIMENT SUMMARY")
        print("="*120)
        print(f"✅ RQ1: Main Comparison - {len(datasets)} datasets, {n_folds}-fold CV")
        print(f"✅ RQ2: Simplex Analysis - {datasets[0]} (single run)")
        print(f"✅ RQ3: Ablation Study - {datasets[0]} (single run)")
        print(f"✅ RQ4: Efficiency Analysis (from RQ1 CV)")
        if len(datasets) > 1:
            print(f"✅ RQ5: Statistical Significance - {len(datasets)} datasets (using RQ1 CV folds)")
        print(f"✅ RQ6: Feature Importance - {datasets[0]} (single run)")
        print("="*120 + "\n")
            


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive GCf-IG Experiments")
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    # ✅ Patch: default to all four datasets
    parser.add_argument('--datasets', nargs='+', default=['german', 'compas', 'adult', 'bank'],
                       choices=['german', 'compas', 'adult', 'bank'],
                       help='Datasets to use')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (fewer epochs and methods)')
    parser.add_argument('--rq', type=str, default='all',
                       choices=['all', 'rq1', 'rq2', 'rq3', 'rq4', 'rq5', 'rq6'],
                       help='Which research question to run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    # After line 1662 (after --seed argument):
    parser.add_argument('--tune', action='store_true',
                   help='Run hyperparameter tuning before experiments')
    parser.add_argument('--tune-trials', type=int, default=20,
                   help='Number of tuning trials (default: 20)')
    parser.add_argument('--tune-dataset', type=str, default=None,
                   help='Dataset to tune on (default: first in datasets list)')
    
    args = parser.parse_args()
    
    # Create experiment
    experiment = ComprehensiveExperiment(
        output_dir=args.output_dir,
        random_seed=args.seed
    )
    
    # Run experiments
    if args.rq == 'all':
        # STEP 1: Hyperparameter tuning (if requested)
        if args.tune:
            tune_dataset = args.tune_dataset if args.tune_dataset else args.datasets[0]
            print("\n" + "="*120)
            print(f"STEP 1: HYPERPARAMETER TUNING on {tune_dataset}")
            print("="*120)
            
            best_params = experiment.tune_all_hyperparameters(
                dataset_name=tune_dataset,
                methods=['unconstrained', 'lagrangian', 'adversarial', 'gcf_ig'],
                n_trials_base=15,
                n_trials_method=args.tune_trials,
                random_state=args.seed
            )
            
            print("\n" + "="*120)
            print("TUNING COMPLETE - Using tuned parameters for evaluation")
            print("="*120)
            
            # Update experiment config with tuned params
            # (We'll use them in run_all_experiments via get_default_config)
            experiment.tuned_params = best_params
        
        # STEP 2: Run all experiments (with tuned params if available)
        experiment.run_all_experiments(args.datasets, quick_mode=args.quick)
        
    elif args.rq == 'rq1':
        experiment.run_rq1_with_kfold_cv(args.datasets)
    elif args.rq == 'rq2':
        experiment.run_rq2_simplex_analysis(args.datasets[0])
    elif args.rq == 'rq3':
        experiment.run_rq3_ablation_study(args.datasets[0])
    elif args.rq == 'rq4':
        experiment.run_rq4_efficiency_analysis()
    elif args.rq == 'rq5':
        experiment.run_rq5_statistical_significance_cv()
    elif args.rq == 'rq6':
        experiment.run_rq6_feature_importance_analysis()


if __name__ == "__main__":
    main()
