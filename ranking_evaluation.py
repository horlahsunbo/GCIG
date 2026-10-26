"""
Ranking-based evaluation for comparing fairness methods.

Implements ranking tables like Table 3 in EDiff paper.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def rank_methods_on_metric(
    results: Dict[str, Dict],
    metric_name: str,
    ascending: bool = True
) -> Dict[str, int]:
    """
    Rank methods on a single metric.
    
    Args:
        results: Dict of {method_name: {'test': metrics_dict}}
        metric_name: Name of metric to rank on
        ascending: True if lower is better (e.g., fairness violations)
        
    Returns:
        ranks: Dict of {method_name: rank} where 1 is best
    """
    # Extract metric values
    values = {}
    for method, result in results.items():
        value = result['test'].get(metric_name, np.inf if ascending else -np.inf)
        values[method] = value
    
    # Sort and assign ranks
    sorted_methods = sorted(values.items(), key=lambda x: x[1], reverse=not ascending)
    
    ranks = {}
    for i, (method, value) in enumerate(sorted_methods):
        ranks[method] = i + 1
    
    return ranks


def rank_all_metrics(
    results: Dict[str, Dict],
    metrics_config: Dict[str, bool]
) -> pd.DataFrame:
    """
    Rank methods on all metrics.
    
    Args:
        results: Method results
        metrics_config: Dict of {metric_name: ascending} where ascending=True means lower is better
        
    Returns:
        df: DataFrame with methods as rows, metrics as columns, values are ranks
    """
    method_names = list(results.keys())
    metric_names = list(metrics_config.keys())
    
    # Create rank matrix
    rank_matrix = np.zeros((len(method_names), len(metric_names)))
    
    for j, metric_name in enumerate(metric_names):
        ascending = metrics_config[metric_name]
        ranks = rank_methods_on_metric(results, metric_name, ascending)
        
        for i, method in enumerate(method_names):
            rank_matrix[i, j] = ranks.get(method, len(method_names))
    
    # Create DataFrame
    df = pd.DataFrame(
        rank_matrix,
        index=method_names,
        columns=metric_names
    )
    
    return df


def compute_rank_statistics(
    rank_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute statistics on ranks (mean, std, etc.).
    
    Args:
        rank_df: DataFrame of ranks
        
    Returns:
        stats_df: DataFrame with mean rank and std error
    """
    stats_df = pd.DataFrame({
        'Mean Rank': rank_df.mean(axis=1),
        'Std Error': rank_df.std(axis=1) / np.sqrt(rank_df.shape[1]),
        'Best Count': (rank_df == 1).sum(axis=1),
        'Worst Count': (rank_df == rank_df.max().max()).sum(axis=1)
    })
    
    # Sort by mean rank
    stats_df = stats_df.sort_values('Mean Rank')
    
    return stats_df


def create_ranking_table(
    results: Dict[str, Dict],
    metrics_config: Dict[str, bool] = None,
    dataset_name: str = "Dataset"
) -> pd.DataFrame:
    """
    Create complete ranking table (like Table 3 in EDiff).
    
    Args:
        results: Method results
        metrics_config: Metric configurations (if None, uses defaults)
        dataset_name: Name of dataset for display
        
    Returns:
        table: Formatted ranking table
    """
    # Default metrics configuration
    if metrics_config is None:
        metrics_config = {
            'accuracy': False,  # Higher is better
            'f1': False,
            'auc': False,
            'statistical_parity': True,  # Lower is better
            'equalized_odds': True,
            'gcf_ig': True,
        }
    
    # Compute ranks
    rank_df = rank_all_metrics(results, metrics_config)
    
    # Add mean rank column
    rank_df['Mean Rank'] = rank_df.mean(axis=1)
    
    # Sort by mean rank
    rank_df = rank_df.sort_values('Mean Rank')
    
    # Format for display
    rank_df = rank_df.round(2)
    
    return rank_df


def aggregate_ranks_across_datasets(
    dataset_results: Dict[str, Dict[str, Dict]],
    metrics_config: Dict[str, bool] = None
) -> pd.DataFrame:
    """
    Aggregate rankings across multiple datasets.
    
    Args:
        dataset_results: Dict of {dataset_name: {method_name: results}}
        metrics_config: Metric configurations
        
    Returns:
        agg_table: Aggregated ranking table with mean ± std across datasets
    """
    if metrics_config is None:
        metrics_config = {
            'accuracy': False,
            'f1': False,
            'auc': False,
            'statistical_parity': True,
            'equalized_odds': True,
            'gcf_ig': True,
        }
    
    # Get all methods (assume same across datasets)
    first_dataset = list(dataset_results.values())[0]
    method_names = list(first_dataset.keys())
    metric_names = list(metrics_config.keys())
    
    # Collect ranks for each dataset
    all_ranks = {method: {metric: [] for metric in metric_names} 
                 for method in method_names}
    
    for dataset_name, results in dataset_results.items():
        rank_df = rank_all_metrics(results, metrics_config)
        
        for method in method_names:
            if method in rank_df.index:
                for metric in metric_names:
                    if metric in rank_df.columns:
                        all_ranks[method][metric].append(rank_df.loc[method, metric])
    
    # Compute mean and std
    agg_data = []
    for method in method_names:
        row = {'Method': method}
        for metric in metric_names:
            ranks = all_ranks[method][metric]
            if len(ranks) > 0:
                mean_rank = np.mean(ranks)
                std_err = np.std(ranks) / np.sqrt(len(ranks))
                row[metric] = f"{mean_rank:.2f} ± {std_err:.2f}"
            else:
                row[metric] = "N/A"
        agg_data.append(row)
    
    agg_df = pd.DataFrame(agg_data)
    
    # Add overall mean rank
    mean_ranks = []
    for method in method_names:
        all_method_ranks = []
        for metric_ranks in all_ranks[method].values():
            all_method_ranks.extend(metric_ranks)
        if len(all_method_ranks) > 0:
            mean_ranks.append(np.mean(all_method_ranks))
        else:
            mean_ranks.append(np.inf)
    
    agg_df['Mean Rank'] = mean_ranks
    agg_df = agg_df.sort_values('Mean Rank')
    
    return agg_df


def print_ranking_table(
    rank_df: pd.DataFrame,
    title: str = "Method Rankings"
):
    """Pretty print ranking table."""
    print("\n" + "="*120)
    print(f"{title:^120}")
    print("="*120)
    print("\nNote: Lower rank is better (1 = best)")
    print("\n" + rank_df.to_string())
    print("\n" + "="*120 + "\n")


# ============================================================================
# NEW: Statistical Significance Testing
# ============================================================================

def wilcoxon_test_across_datasets(
    dataset_results: Dict[str, Dict[str, Dict]],
    metric: str,
    method1: str,
    method2: str
) -> Tuple[float, float]:
    """
    Compare two methods across multiple datasets using Wilcoxon signed-rank test.
    
    Args:
        dataset_results: Dict of {dataset_name: {method_name: results}}
        metric: Metric name to compare
        method1: First method name
        method2: Second method name
        
    Returns:
        statistic: Test statistic
        p_value: P-value
    """
    values1 = []
    values2 = []
    
    for dataset_name, results in dataset_results.items():
        if method1 in results and method2 in results:
            val1 = results[method1]['test'].get(metric, np.nan)
            val2 = results[method2]['test'].get(metric, np.nan)
            
            if not np.isnan(val1) and not np.isnan(val2):
                values1.append(val1)
                values2.append(val2)
    
    if len(values1) < 2:
        return 0.0, 1.0
    
    try:
        statistic, p_value = stats.wilcoxon(values1, values2)
        return statistic, p_value
    except:
        return 0.0, 1.0


def statistical_significance_test(
    results: Dict[str, Dict],
    metric_name: str,
    method1: str,
    method2: str,
    test: str = 'wilcoxon'
) -> Tuple[float, float]:
    """
    Test if difference between two methods is statistically significant.
    
    Args:
        results: Results from multiple runs/seeds
        metric_name: Metric to compare
        method1, method2: Method names
        test: 'wilcoxon' or 't-test'
        
    Returns:
        statistic, p_value
    """
    # This assumes results contain multiple runs
    # For now, return placeholder
    # In full implementation, you'd track multiple seeds
    
    values1 = [results[method1]['test'][metric_name]]
    values2 = [results[method2]['test'][metric_name]]
    
    if test == 'wilcoxon' and len(values1) > 1:
        statistic, p_value = stats.wilcoxon(values1, values2)
    elif test == 't-test' and len(values1) > 1:
        statistic, p_value = stats.ttest_rel(values1, values2)
    else:
        return 0.0, 1.0
    
    return statistic, p_value


def compute_pairwise_significance_matrix(
    dataset_results: Dict[str, Dict[str, Dict]],
    metric: str,
    method_names: List[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute pairwise statistical significance matrix for all methods.
    
    Args:
        dataset_results: Dict of {dataset_name: {method_name: results}}
        metric: Metric to compare
        method_names: List of method names
        alpha: Significance level
        
    Returns:
        sig_matrix: DataFrame with p-values and significance markers
    """
    n_methods = len(method_names)
    p_matrix = np.ones((n_methods, n_methods))
    
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i != j:
                _, p_value = wilcoxon_test_across_datasets(
                    dataset_results, metric, method1, method2
                )
                p_matrix[i, j] = p_value
    
    sig_matrix = pd.DataFrame(
        p_matrix,
        index=method_names,
        columns=method_names
    )
    
    return sig_matrix


def create_significance_heatmap(
    sig_matrix: pd.DataFrame,
    metric_name: str,
    alpha: float = 0.05,
    save_path: str = None
):
    """
    Visualize statistical significance matrix as heatmap.
    
    Args:
        sig_matrix: P-value matrix from compute_pairwise_significance_matrix
        metric_name: Name of metric being compared
        alpha: Significance level
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create annotation matrix with significance markers
    annot_matrix = sig_matrix.copy()
    for i in range(len(sig_matrix)):
        for j in range(len(sig_matrix.columns)):
            p_val = sig_matrix.iloc[i, j]
            if i == j:
                annot_matrix.iloc[i, j] = "-"
            elif p_val < 0.001:
                annot_matrix.iloc[i, j] = f"{p_val:.4f}***"
            elif p_val < 0.01:
                annot_matrix.iloc[i, j] = f"{p_val:.4f}**"
            elif p_val < alpha:
                annot_matrix.iloc[i, j] = f"{p_val:.4f}*"
            else:
                annot_matrix.iloc[i, j] = f"{p_val:.4f}"
    
    # Create heatmap
    sns.heatmap(
        sig_matrix,
        annot=annot_matrix,
        fmt='',
        cmap='RdYlGn',
        center=alpha,
        vmin=0,
        vmax=0.1,
        cbar_kws={'label': 'p-value'},
        ax=ax
    )
    
    ax.set_title(f'Statistical Significance Matrix: {metric_name}\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel('Method', fontsize=11, fontweight='bold')
    ax.set_ylabel('Method', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved significance heatmap to {save_path}")
    
    plt.show()
    plt.close()


def generate_significance_report(
    dataset_results: Dict[str, Dict[str, Dict]],
    reference_method: str = 'gcf_ig',
    metrics: List[str] = None,
    save_dir: str = None
):
    """
    Generate statistical significance report comparing all methods to reference.
    
    Args:
        dataset_results: Dict of {dataset_name: {method_name: results}}
        reference_method: Method to compare against (typically your method)
        metrics: List of metrics to test (if None, uses defaults)
        save_dir: Directory to save outputs
    """
    if metrics is None:
        metrics = ['accuracy', 'f1', 'statistical_parity', 'equalized_odds', 'gcf_ig']
    
    method_names = list(list(dataset_results.values())[0].keys())
    other_methods = [m for m in method_names if m != reference_method]
    
    print("\n" + "="*120)
    print(f"STATISTICAL SIGNIFICANCE REPORT: {reference_method.upper()} vs. BASELINES")
    print("="*120)
    
    results_table = []
    
    for metric in metrics:
        print(f"\n\nMetric: {metric.upper()}")
        print("-" * 80)
        
        for other_method in other_methods:
            stat, p_value = wilcoxon_test_across_datasets(
                dataset_results, metric, reference_method, other_method
            )
            
            # Determine significance level
            if p_value < 0.001:
                sig_marker = "***"
            elif p_value < 0.01:
                sig_marker = "**"
            elif p_value < 0.05:
                sig_marker = "*"
            else:
                sig_marker = "n.s."
            
            print(f"  {reference_method} vs. {other_method:<20} | p={p_value:.4f} {sig_marker}")
            
            results_table.append({
                'Metric': metric,
                'Comparison': f'{reference_method} vs. {other_method}',
                'Statistic': stat,
                'P-value': p_value,
                'Significant': sig_marker
            })
    
    # Save table
    if save_dir:
        df = pd.DataFrame(results_table)
        df.to_csv(f"{save_dir}/significance_tests.csv", index=False)
        print(f"\n\nSaved significance test results to {save_dir}/significance_tests.csv")
    
    print("\n" + "="*120)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print("="*120 + "\n")


# ============================================================================
# NEW: Feature Importance Visualization (Figure 7 from EDiff)
# ============================================================================

def compute_feature_attributions(
    model,
    data,
    baselines: Dict[Tuple[int, int], np.ndarray],
    n_steps: int = 32,
    device: str = 'cpu'
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute mean feature attributions for each (label, group) combination.
    
    Args:
        model: Trained model
        data: FairnessDataset
        baselines: Baseline vectors
        n_steps: Integration steps
        device: Device
        
    Returns:
        mean_attributions: Dict of {(label, group): mean_attribution_vector}
    """
    import torch
    
    # Handle different model types
    if hasattr(model, 'model'):
        torch_model = model.model
    else:
        torch_model = model
    
    # Check if it's a linear model (sklearn)
    if hasattr(torch_model, 'coef_'):
        # For linear models, IG is just w * (x - baseline)
        w = np.asarray(torch_model.coef_).reshape(-1)
        
        mean_attributions = {}
        for label in [0, 1]:
            for group in [0, 1]:
                idx = np.where((data.y == label) & (data.a == group))[0]
                if len(idx) == 0:
                    mean_attributions[(label, group)] = np.zeros_like(w)
                    continue
                
                X_subset = data.X[idx]
                baseline = baselines[(label, group)]
                
                # IG for linear: w * (x - baseline)
                ig = (X_subset - baseline) * w
                mean_attributions[(label, group)] = np.abs(ig).mean(axis=0)
        
        return mean_attributions
    
    # For neural networks
    torch_model.eval()
    torch_model.to(device)
    
    mean_attributions = {}
    
    for label in [0, 1]:
        for group in [0, 1]:
            idx = np.where((data.y == label) & (data.a == group))[0]
            
            if len(idx) == 0:
                mean_attributions[(label, group)] = np.zeros(data.X.shape[1])
                continue
            
            X_subset = data.X[idx]
            baseline = baselines[(label, group)]
            
            # Compute IG for this subset
            attributions_list = []
            
            for i in range(len(X_subset)):
                x = X_subset[i:i+1]
                
                # Integrated gradients
                grad_sum = np.zeros_like(x)
                
                for t in range(1, n_steps + 1):
                    alpha = t / n_steps
                    x_t = baseline + alpha * (x - baseline)
                    x_t_tensor = torch.from_numpy(x_t).float().to(device).requires_grad_(True)
                    
                    logit = torch_model(x_t_tensor)
                    logit.backward()
                    
                    grad = x_t_tensor.grad.cpu().numpy()
                    grad_sum += grad
                
                ig = (x - baseline) * (grad_sum / n_steps)
                attributions_list.append(np.abs(ig).flatten())
            
            mean_attributions[(label, group)] = np.mean(attributions_list, axis=0)
    
    return mean_attributions


def plot_feature_importance_comparison(
    model1,
    model2,
    data,
    baselines,
    feature_names: List[str],
    model1_name: str = "Method 1",
    model2_name: str = "Method 2",
    top_k: int = 10,
    save_path: str = None
):
    """
    Compare feature importance between two models (like Figure 7 in EDiff).
    
    Shows side-by-side heatmaps of mean feature attributions per (label, group).
    
    Args:
        model1: First model (e.g., best performing)
        model2: Second model (e.g., worst performing or baseline)
        data: FairnessDataset
        baselines: Baseline vectors
        feature_names: List of feature names
        model1_name: Name for model 1
        model2_name: Name for model 2
        top_k: Number of top features to show
        save_path: Path to save figure
    """
    # Compute attributions
    attr1 = compute_feature_attributions(model1, data, baselines)
    attr2 = compute_feature_attributions(model2, data, baselines)
    
    # Find features with largest disparities in either model
    disparities1 = np.abs(attr1[(0, 0)] - attr1[(0, 1)]) + np.abs(attr1[(1, 0)] - attr1[(1, 1)])
    disparities2 = np.abs(attr2[(0, 0)] - attr2[(0, 1)]) + np.abs(attr2[(1, 0)] - attr2[(1, 1)])
    
    max_disparities = np.maximum(disparities1, disparities2)
    top_indices = np.argsort(max_disparities)[-top_k:][::-1]
    
    selected_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
                        for i in top_indices]
    
    # Create data matrices for heatmap
    data1 = np.array([
        [attr1[(0, 0)][i], attr1[(0, 1)][i]],
        [attr1[(1, 0)][i], attr1[(1, 1)][i]]
    ] for i in top_indices).T
    
    data2 = np.array([
        [attr2[(0, 0)][i], attr2[(0, 1)][i]],
        [attr2[(1, 0)][i], attr2[(1, 1)][i]]
    ] for i in top_indices).T
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    vmax = max(np.max(data1), np.max(data2))
    
    # Model 1
    sns.heatmap(
        data1,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        yticklabels=['Group 0', 'Group 1'],
        xticklabels=selected_features,
        ax=axes[0],
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'Mean |Attribution|'}
    )
    axes[0].set_title(f'{model1_name}\n(Lower GCf-IG = More Fair)', 
                     fontsize=12, fontweight='bold')
    axes[0].set_ylabel('(Label 0 | Label 1)', fontsize=11, fontweight='bold')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Model 2
    sns.heatmap(
        data2,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        yticklabels=['Group 0', 'Group 1'],
        xticklabels=selected_features,
        ax=axes[1],
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'Mean |Attribution|'}
    )
    axes[1].set_title(f'{model2_name}\n(Higher GCf-IG = Less Fair)', 
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('(Label 0 | Label 1)', fontsize=11, fontweight='bold')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Feature Attribution Comparison: Top Features with Largest Group Disparities',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance comparison to {save_path}")
    
    plt.show()
    plt.close()


def create_disparity_barplot(
    model,
    data,
    baselines,
    feature_names: List[str],
    model_name: str = "Model",
    top_k: int = 10,
    save_path: str = None
):
    """
    Create bar plot showing feature attribution disparities between groups.
    
    Args:
        model: Trained model
        data: FairnessDataset
        baselines: Baseline vectors
        feature_names: List of feature names
        model_name: Model name for title
        top_k: Number of features to show
        save_path: Path to save figure
    """
    # Compute attributions
    attr = compute_feature_attributions(model, data, baselines)
    
    # Compute disparities per feature
    disparities = []
    for i in range(len(feature_names)):
        # Average disparity across labels
        disp_label0 = np.abs(attr[(0, 0)][i] - attr[(0, 1)][i])
        disp_label1 = np.abs(attr[(1, 0)][i] - attr[(1, 1)][i])
        avg_disp = (disp_label0 + disp_label1) / 2
        disparities.append(avg_disp)
    
    disparities = np.array(disparities)
    
    # Get top-k
    top_indices = np.argsort(disparities)[-top_k:][::-1]
    top_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
                   for i in top_indices]
    top_disparities = disparities[top_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn_r(top_disparities / top_disparities.max())
    bars = ax.barh(top_features, top_disparities, color=colors)
    
    ax.set_xlabel('Mean Attribution Disparity Between Groups', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Attribution Disparities: {model_name}\n(Higher = More Unfair)', 
                fontsize=13, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_disparities)):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
               va='center', ha='left', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved disparity barplot to {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# Existing Functions
# ============================================================================

def create_comparison_heatmap(
    rank_df: pd.DataFrame,
    save_path: str = None
):
    """
    Create heatmap visualization of rankings.
    
    Args:
        rank_df: Ranking DataFrame
        save_path: Path to save figure
    """
    # Extract numeric columns (exclude Mean Rank)
    numeric_cols = [col for col in rank_df.columns if col != 'Mean Rank']
    plot_data = rank_df[numeric_cols]
    
    fig, ax = plt.subplots(figsize=(12, len(rank_df) * 0.5))
    
    # Create heatmap (lower values = darker = better)
    sns.heatmap(
        plot_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',  # Red for high ranks (bad), green for low ranks (good)
        cbar_kws={'label': 'Rank (1=best)'},
        ax=ax,
        vmin=1,
        vmax=plot_data.max().max()
    )
    
    ax.set_title('Method Rankings Across Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Methods', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ranking heatmap to {save_path}")
    
    plt.show()
    plt.close()


def generate_ranking_report(
    dataset_results: Dict[str, Dict[str, Dict]],
    save_dir: str = None
):
    """
    Generate complete ranking report across datasets.
    
    Args:
        dataset_results: Dict of {dataset_name: {method_name: results}}
        save_dir: Directory to save outputs
    """
    print("\n" + "="*120)
    print("RANKING-BASED EVALUATION REPORT")
    print("="*120)
    
    # Per-dataset rankings
    for dataset_name, results in dataset_results.items():
        print(f"\n\nDataset: {dataset_name}")
        rank_df = create_ranking_table(results, dataset_name=dataset_name)
        print_ranking_table(rank_df, title=f"{dataset_name} - Method Rankings")
        
        if save_dir:
            # Save table as CSV
            rank_df.to_csv(f"{save_dir}/rankings_{dataset_name}.csv")
            
            # Save heatmap
            create_comparison_heatmap(
                rank_df,
                save_path=f"{save_dir}/heatmap_{dataset_name}.png"
            )
    
    # Aggregated rankings
    print("\n\n" + "="*120)
    print("AGGREGATED RANKINGS ACROSS ALL DATASETS")
    print("="*120)
    
    agg_df = aggregate_ranks_across_datasets(dataset_results)
    print("\n" + agg_df.to_string())
    
    if save_dir:
        agg_df.to_csv(f"{save_dir}/rankings_aggregated.csv")
    
    print("\n" + "="*120)