"""
Multi-objective simplex analysis for fairness trade-offs.

Implements the analysis from EDiff paper (Figure 4, Table 4):
- Sample weight combinations across the 2D simplex
- Determine best method for each weight combination
- Visualize preference regions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import Dict, List, Tuple
import itertools


def sample_simplex_2d(n_samples: int = 1000) -> np.ndarray:
    """
    Uniformly sample from 2D simplex (triangle).
    
    For 3 objectives (w1, w2, w3) with w1 + w2 + w3 = 1:
    We sample (w1, w2) and compute w3 = 1 - w1 - w2
    
    Returns:
        weights: (n_samples, 3) array of weight combinations
    """
    weights = []
    
    # Uniform sampling from triangle
    for _ in range(n_samples):
        # Sample two uniform random variables
        u1, u2 = np.random.uniform(0, 1, 2)
        
        # Transform to ensure they're in the simplex
        if u1 + u2 > 1:
            u1 = 1 - u1
            u2 = 1 - u2
        
        w1 = u1
        w2 = u2
        w3 = 1 - w1 - w2
        
        weights.append([w1, w2, w3])
    
    return np.array(weights)


def compute_weighted_score(
    metrics: Dict[str, float],
    weights: np.ndarray,
    objective_names: List[str],
    maximize: Dict[str, bool] = None
) -> float:
    """
    Compute weighted score for given metrics and weights.
    
    Args:
        metrics: Dict of metric values
        weights: Array of weights (3,)
        objective_names: List of 3 objective names (e.g., ['accuracy', 'sp', 'gcf_ig'])
        maximize: Dict indicating whether to maximize each metric (default: True for accuracy, False for fairness)
        
    Returns:
        score: Weighted score (higher is better)
    """
    if maximize is None:
        # Default: maximize accuracy, minimize fairness violations
        maximize = {
            'accuracy': True,
            'f1': True,
            'auc': True,
            'statistical_parity': False,
            'equalized_odds': False,
            'gcf_ig': False,
            'ediff': False
        }
    
    score = 0.0
    for i, obj_name in enumerate(objective_names):
        value = metrics.get(obj_name, 0.0)
        
        # Flip sign if we want to minimize
        if not maximize.get(obj_name, True):
            value = -value
        
        score += weights[i] * value
    
    return score


def find_best_methods_simplex(
    results: Dict[str, Dict],
    objective_names: List[str],
    n_samples: int = 5000,
    maximize: Dict[str, bool] = None
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Find which method is best for each weight combination.
    
    Args:
        results: Dict of {method_name: {'test': metrics_dict}}
        objective_names: List of 3 objectives to optimize
        n_samples: Number of simplex samples
        maximize: Which objectives to maximize vs minimize
        
    Returns:
        best_methods: Dict mapping weight_idx -> method_name
        weights: Sampled weight combinations (n_samples, 3)
        scores: Scores for each method at each weight (n_samples, n_methods)
    """
    # Sample simplex
    weights = sample_simplex_2d(n_samples)
    
    # Get method names
    method_names = list(results.keys())
    n_methods = len(method_names)
    
    # Compute scores for each method at each weight
    scores = np.zeros((n_samples, n_methods))
    
    for i, w in enumerate(weights):
        for j, method_name in enumerate(method_names):
            metrics = results[method_name]['test']
            score = compute_weighted_score(metrics, w, objective_names, maximize)
            scores[i, j] = score
    
    # Find best method for each weight
    best_method_indices = np.argmax(scores, axis=1)
    best_methods = {
        i: method_names[best_method_indices[i]]
        for i in range(n_samples)
    }
    
    return best_methods, weights, scores


def compute_simplex_coverage(
    best_methods: Dict[int, str],
    method_names: List[str]
) -> Dict[str, float]:
    """
    Compute what fraction of simplex each method dominates.
    
    Args:
        best_methods: Dict mapping weight_idx -> method_name
        method_names: List of all method names
        
    Returns:
        coverage: Dict of {method_name: coverage_fraction}
    """
    n_total = len(best_methods)
    
    coverage = {}
    for method in method_names:
        count = sum(1 for m in best_methods.values() if m == method)
        coverage[method] = count / n_total
    
    return coverage


def plot_simplex_triangle(
    best_methods: Dict[int, str],
    weights: np.ndarray,
    objective_names: List[str],
    method_names: List[str],
    save_path: str = None,
    title: str = "Best Method by Preference"
):
    """
    Plot 2D simplex showing which method is best in each region.
    
    Args:
        best_methods: Dict mapping weight_idx -> method_name
        weights: Sampled weights (n_samples, 3)
        objective_names: Names of 3 objectives
        method_names: List of method names
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))
    method_to_color = {name: colors[i] for i, name in enumerate(method_names)}
    
    # Convert 3D simplex coordinates to 2D for plotting
    # Use barycentric coordinates
    # Point (w1, w2, w3) maps to 2D as:
    # x = w2 + 0.5 * w3
    # y = (sqrt(3)/2) * w3
    
    x_coords = []
    y_coords = []
    point_colors = []
    
    for i, w in enumerate(weights):
        w1, w2, w3 = w
        x = w2 + 0.5 * w3
        y = (np.sqrt(3) / 2) * w3
        
        x_coords.append(x)
        y_coords.append(y)
        
        method = best_methods[i]
        point_colors.append(method_to_color[method])
    
    # Scatter plot
    ax.scatter(x_coords, y_coords, c=point_colors, s=5, alpha=0.6)
    
    # Draw triangle boundary
    triangle = Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3)/2]],
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(triangle)
    
    # Add labels at corners
    ax.text(-0.05, -0.05, objective_names[0], ha='right', va='top', fontsize=12, fontweight='bold')
    ax.text(1.05, -0.05, objective_names[1], ha='left', va='top', fontsize=12, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, objective_names[2], ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=method_to_color[m], 
               markersize=10, label=m)
        for m in method_names
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved simplex plot to {save_path}")
    
    plt.show()


def run_simplex_analysis(
    results: Dict[str, Dict],
    objective_names: List[str] = ['accuracy', 'statistical_parity', 'gcf_ig'],
    n_samples: int = 5000,
    save_path: str = None,
    title: str = None
) -> Dict:
    """
    Complete simplex analysis pipeline.
    
    Args:
        results: Dict of {method_name: {'test': metrics_dict}}
        objective_names: 3 objectives to trade off
        n_samples: Simplex samples
        save_path: Path to save plot
        title: Plot title
        
    Returns:
        analysis: Dict with coverage statistics
    """
    print("\n" + "="*80)
    print("MULTI-OBJECTIVE SIMPLEX ANALYSIS")
    print("="*80)
    print(f"\nObjectives: {objective_names}")
    print(f"Sampling {n_samples} weight combinations...")
    
    # Find best methods
    best_methods, weights, scores = find_best_methods_simplex(
        results, objective_names, n_samples
    )
    
    # Compute coverage
    method_names = list(results.keys())
    coverage = compute_simplex_coverage(best_methods, method_names)
    
    print("\n" + "="*80)
    print("SIMPLEX COVERAGE (% of preference space dominated)")
    print("="*80)
    for method, cov in sorted(coverage.items(), key=lambda x: -x[1]):
        print(f"  {method:<30} {cov*100:>6.2f}%")
    
    # Plot
    if title is None:
        title = f"Best Method by User Preference\n({', '.join(objective_names)})"
    
    plot_simplex_triangle(
        best_methods, weights, objective_names, method_names,
        save_path=save_path, title=title
    )
    
    analysis = {
        'coverage': coverage,
        'best_methods': best_methods,
        'weights': weights,
        'scores': scores
    }
    
    return analysis


def compare_multiple_objectives(
    results: Dict[str, Dict],
    objective_sets: List[Tuple[str, str, str]],
    save_dir: str = None
):
    """
    Run simplex analysis for multiple objective combinations.
    
    Args:
        results: Method results
        objective_sets: List of (obj1, obj2, obj3) tuples
        save_dir: Directory to save plots
    """
    all_analyses = {}
    
    for obj_set in objective_sets:
        print(f"\n{'='*80}")
        print(f"Analyzing objectives: {obj_set}")
        print(f"{'='*80}")
        
        save_path = None
        if save_dir:
            obj_str = "_".join(obj_set)
            save_path = f"{save_dir}/simplex_{obj_str}.png"
        
        analysis = run_simplex_analysis(
            results, 
            objective_names=list(obj_set),
            save_path=save_path
        )
        
        all_analyses[obj_set] = analysis
    
    return all_analyses
