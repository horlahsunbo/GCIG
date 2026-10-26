# ===== FILE: metrics.py =====
"""
Evaluation metrics for fairness experiments.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from typing import Dict, Tuple


def build_label_group_baselines(
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    method: str = "mean",
) -> Dict[Tuple[int, int], np.ndarray]:
    """Build (label, group) baselines b_{y,g} from arrays.

    Keys are (y, g) with y,g in {0,1}. If a (y,g) bucket is empty, we fall
    back to the label-conditional mean/median.
    """
    baselines: Dict[Tuple[int, int], np.ndarray] = {}
    for label in [0, 1]:
        idx_label = np.where(y == label)[0]
        if len(idx_label) == 0:
            # If a label is missing entirely, fall back to overall.
            idx_label = np.arange(len(y))
        if method == "mean":
            b_label = X[idx_label].mean(axis=0)
        elif method == "median":
            b_label = np.median(X[idx_label], axis=0)
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        for group in [0, 1]:
            idx = np.where((y == label) & (a == group))[0]
            if len(idx) == 0:
                baselines[(label, group)] = b_label.copy()
            else:
                if method == "mean":
                    baselines[(label, group)] = X[idx].mean(axis=0)
                else:
                    baselines[(label, group)] = np.median(X[idx], axis=0)

    return baselines


def compute_performance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute standard classification performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = 0.0
    
    return metrics


def compute_statistical_parity(
    y_pred: np.ndarray,
    a: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Statistical Parity (SP) difference.
    SP = |P(ŷ=1|a=0) - P(ŷ=1|a=1)|
    
    Args:
        y_pred: Predicted labels
        a: Protected attributes
        
    Returns:
        sp_diff: Statistical parity difference
        details: Per-group positive rates
    """
    # FIXED: Binarize predictions first
    if y_pred.dtype in [np.float32, np.float64]:
        y_pred_binary = (y_pred >= 0.5).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    mask_0 = (a == 0)
    mask_1 = (a == 1)
    
    # Use binarized predictions, not probabilities
    pr_0 = float(y_pred_binary[mask_0].mean()) if mask_0.sum() > 0 else 0.0
    pr_1 = float(y_pred_binary[mask_1].mean()) if mask_1.sum() > 0 else 0.0
    
    sp_diff = abs(pr_0 - pr_1)
    
    details = {
        'pr_group0': pr_0,
        'pr_group1': pr_1,
        'sp_diff': sp_diff
    }
    
    return sp_diff, details


def compute_equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Equalized Odds (EO) difference.
    EO = |TPR_0 - TPR_1| + |FPR_0 - FPR_1|
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        a: Protected attributes
        
    Returns:
        eo_diff: Equalized odds difference
        details: Per-group TPR/FPR
    """
    groups = [0, 1]
    tpr = {}
    fpr = {}
    
    for g in groups:
        mask_g = (a == g)
        
        # Get group-specific data
        y_true_g = y_true[mask_g]
        y_pred_g = y_pred[mask_g]
        
        # FIXED: Binarize predictions
        if y_pred_g.dtype in [np.float32, np.float64]:
            y_pred_binary_g = (y_pred_g >= 0.5).astype(int)
        else:
            y_pred_binary_g = y_pred_g.astype(int)
        
        # FIXED: Use confusion matrix, not averages
        # Calculate TPR = TP / (TP + FN)
        actual_positives = (y_true_g == 1)
        if actual_positives.sum() > 0:
            tp = ((y_true_g == 1) & (y_pred_binary_g == 1)).sum()
            fn = ((y_true_g == 1) & (y_pred_binary_g == 0)).sum()
            tpr[g] = float(tp) / float(tp + fn)
        else:
            tpr[g] = 0.0
        
        # Calculate FPR = FP / (FP + TN)
        actual_negatives = (y_true_g == 0)
        if actual_negatives.sum() > 0:
            fp = ((y_true_g == 0) & (y_pred_binary_g == 1)).sum()
            tn = ((y_true_g == 0) & (y_pred_binary_g == 0)).sum()
            fpr[g] = float(fp) / float(fp + tn)
        else:
            fpr[g] = 0.0
    
    tpr_diff = abs(tpr[0] - tpr[1])
    fpr_diff = abs(fpr[0] - fpr[1])
    
    eo_diff = tpr_diff + fpr_diff
    
    details = {
        'tpr_group0': tpr[0],
        'tpr_group1': tpr[1],
        'fpr_group0': fpr[0],
        'fpr_group1': fpr[1],
        'tpr_diff': tpr_diff,
        'fpr_diff': fpr_diff,
        'eo_diff': eo_diff
    }
    
    return eo_diff, details


def compute_disparate_impact(
    y_pred: np.ndarray,
    a: np.ndarray,
    privileged_group: int = 0
) -> Tuple[float, Dict[str, float]]:
    """
    Compute Disparate Impact (DI) ratio.
    DI = P(ŷ=1|unprivileged) / P(ŷ=1|privileged)
    
    Args:
        y_pred: Predicted labels
        a: Protected attributes
        privileged_group: Which group is privileged (0 or 1)
        
    Returns:
        di_ratio: Disparate impact ratio
        details: Per-group rates
    """
    unprivileged_group = 1 - privileged_group
    
    mask_priv = (a == privileged_group)
    mask_unpriv = (a == unprivileged_group)
    
    pr_priv = y_pred[mask_priv].mean() if mask_priv.sum() > 0 else 0.0
    pr_unpriv = y_pred[mask_unpriv].mean() if mask_unpriv.sum() > 0 else 0.0
    
    di_ratio = pr_unpriv / (pr_priv + 1e-8)
    
    details = {
        'pr_privileged': pr_priv,
        'pr_unprivileged': pr_unpriv,
        'di_ratio': di_ratio
    }
    
    return di_ratio, details


def compute_gcf_ig_metric(
    model,
    X: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    baselines: Dict[Tuple[int, int], np.ndarray],
    n_steps: int = 32,
    norm: str = 'l2',
    device: str = 'cpu'
) -> Tuple[float, Dict]:
    """
    Compute GCf-IG fairness metric (procedural fairness measure).
    
    This is the evaluation metric version (not the loss).
    Measures average difference in IG attributions between groups.
    
    Args:
        model: Trained neural network
        X: Features
        y: Labels
        a: Protected attributes
        baselines: Baseline vectors for each (label, group)
        n_steps: Integration steps
        norm: Normalization norm
        device: Computation device
        
    Returns:
        gcf_ig: Overall GCf-IG score (lower is more fair)
        details: Per-label scores
    """
    """GCf-IG evaluation metric.

    Supports two cases:
    1) PyTorch models: computes GCf-IG via the differentiable GCfIGLoss but
       with create_graph=False (faster, no higher-order graphs).
    2) Linear sklearn models with coef_: uses closed-form IG for linear logits:
       IG(x; b) = w ⊙ (x - b).
    """
    import torch
    from losses import GCfIGLoss

    # Case: sklearn-style linear model (e.g., LogisticRegression)
    if hasattr(model, "coef_"):
        w = np.asarray(model.coef_).reshape(-1)
        eps = 1e-8
        scores = []
        for label in [0, 1]:
            idx = np.where(y == label)[0]
            if len(idx) == 0:
                continue
            X_l = X[idx]
            B0 = baselines[(label, 0)]
            B1 = baselines[(label, 1)]
            ig0 = (X_l - B0) * w
            ig1 = (X_l - B1) * w
            if norm == "l1":
                ig0 = ig0 / (np.abs(ig0).sum(axis=1, keepdims=True) + eps)
                ig1 = ig1 / (np.abs(ig1).sum(axis=1, keepdims=True) + eps)
            else:
                ig0 = ig0 / (np.sqrt((ig0 ** 2).sum(axis=1, keepdims=True)) + eps)
                ig1 = ig1 / (np.sqrt((ig1 ** 2).sum(axis=1, keepdims=True)) + eps)
            diff = ig0 - ig1
            v = 0.5 * np.sqrt((diff ** 2).sum(axis=1))
            scores.append(v.mean())
        return float(np.mean(scores)) if scores else 0.0, {}

    # Case: PyTorch model or trainer-like wrapper
    torch_model = model.model if hasattr(model, "model") else model
    torch_model.eval()

    gcf_ig_loss_fn = GCfIGLoss(
        n_integration_steps=n_steps,
        norm=norm,
        stratify_by_label=True,
        create_graph=False,
    )

    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y).long().to(device)
    a_tensor = torch.from_numpy(a).long().to(device)
    baselines_tensor = {k: torch.from_numpy(v).float().to(device) for k, v in baselines.items()}

    # Need gradients w.r.t. inputs to compute IG
    with torch.enable_grad():
        gcf_ig_value = gcf_ig_loss_fn(torch_model, X_tensor, y_tensor, a_tensor, baselines_tensor)

    return float(gcf_ig_value.detach().cpu().item()), {}


def evaluate_model(
    model,
    data,
    baselines: Dict = None,
    device: str = 'cpu',
    compute_gcf_ig: bool = False
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a model.
    
    Args:
        model: Trained model (can be sklearn or PyTorch)
        data: FairnessDataset
        baselines: Baselines for GCf-IG computation
        device: Device for computation
        compute_gcf_ig: Whether to compute GCf-IG metric
        
    Returns:
        metrics: Dictionary of all metrics
    """
    def _predict_proba_any(m, X, a=None):
        """Return probabilities in [0,1] for the positive class."""
        # Our BaselineMethod API
        if hasattr(m, "predict_proba"):
            try:
                return m.predict_proba(X, a)
            except TypeError:
                return m.predict_proba(X)

        # sklearn estimator
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(X)
            return p[:, 1] if p.ndim == 2 else p

        # PyTorch module
        import torch
        tm = m.model if hasattr(m, "model") else m
        tm.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(device)
            logits = tm(X_tensor)
            return torch.sigmoid(logits).cpu().numpy()

    # Get predictions (some post-processing methods need sensitive features)
    y_prob = _predict_proba_any(model, data.X, getattr(data, "a", None))
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Performance metrics
    perf_metrics = compute_performance_metrics(data.y, y_pred, y_prob)
    
    # Fairness metrics
    sp_diff, sp_details = compute_statistical_parity(y_pred, data.a)
    eo_diff, eo_details = compute_equalized_odds(data.y, y_pred, data.a)
    
    metrics = {
        **perf_metrics,
        'statistical_parity': sp_diff,
        'equalized_odds': eo_diff,
        **sp_details,
        **eo_details
    }
    
    # GCf-IG metric
    if compute_gcf_ig and baselines is not None:
        # If a method exposes a differentiable explanation model, prefer it.
        expl_model = getattr(model, "explanation_model", model)
        gcf_ig_value, _ = compute_gcf_ig_metric(
            expl_model, data.X, data.y, data.a, baselines, device=device
        )
        metrics['gcf_ig'] = gcf_ig_value
        
    # ---- add these for bootstrap + later statistical tests ----
    metrics["_y_prob"] = y_prob
    metrics["_y_pred"] = y_pred
    metrics["_y_true"] = data.y
    metrics["_a"] = data.a
    
    return metrics

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

def bootstrap_metrics(y_true, y_prob, a, n_boot=500, seed=0):
    """
    Bootstrap CI over test samples. Uses fixed model predictions and resamples indices.
    Returns dict: metric -> (mean, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = {
        "accuracy": [],
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "statistical_parity": [],
        "equalized_odds": [],
        "tpr_g0": [], "tpr_g1": [],
        "fpr_g0": [], "fpr_g1": [],
        "dtpr": [], "dfpr": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        aa = a[idx]
        yhat = (yp >= 0.5).astype(int)

        # performance
        stats["accuracy"].append(accuracy_score(yt, yhat))
        try:
            stats["auc"].append(roc_auc_score(yt, yp))
        except:
            stats["auc"].append(np.nan)
        stats["f1"].append(f1_score(yt, yhat, zero_division=0))
        stats["precision"].append(precision_score(yt, yhat, zero_division=0))
        stats["recall"].append(recall_score(yt, yhat, zero_division=0))

        # SP - FIXED: binarize predictions first
        yhat_binary = (yhat >= 0.5).astype(int) if yhat.dtype in [np.float32, np.float64] else yhat.astype(int)
        pr0 = yhat_binary[aa == 0].mean() if np.any(aa == 0) else np.nan
        pr1 = yhat_binary[aa == 1].mean() if np.any(aa == 1) else np.nan
        stats["statistical_parity"].append(abs(pr0 - pr1))

        # EO components - FIXED
        # Binarize predictions
        yp_binary = (yp >= 0.5).astype(int) if yp.dtype in [np.float32, np.float64] else yp.astype(int)
        
        def _tpr(group_mask):
            """True Positive Rate for group"""
            mask = group_mask & (yt == 1)
            if np.any(mask):
                tp = np.sum((yt[mask] == 1) & (yp_binary[mask] == 1))
                fn = np.sum((yt[mask] == 1) & (yp_binary[mask] == 0))
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0
            return np.nan
        
        def _fpr(group_mask):
            """False Positive Rate for group"""
            mask = group_mask & (yt == 0)
            if np.any(mask):
                fp = np.sum((yt[mask] == 0) & (yp_binary[mask] == 1))
                tn = np.sum((yt[mask] == 0) & (yp_binary[mask] == 0))
                return fp / (fp + tn) if (fp + tn) > 0 else 0.0
            return np.nan

        tpr0 = _tpr(aa == 0)
        tpr1 = _tpr(aa == 1)
        fpr0 = _fpr(aa == 0)
        fpr1 = _fpr(aa == 1)

        stats["tpr_g0"].append(tpr0); stats["tpr_g1"].append(tpr1)
        stats["fpr_g0"].append(fpr0); stats["fpr_g1"].append(fpr1)

        dtpr = abs(tpr0 - tpr1)
        dfpr = abs(fpr0 - fpr1)
        stats["dtpr"].append(dtpr)
        stats["dfpr"].append(dfpr)
        stats["equalized_odds"].append(dtpr + dfpr)

    out = {}
    for k, arr in stats.items():
        arr = np.array(arr, dtype=float)
        m = np.nanmean(arr)
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        out[k] = (m, lo, hi)
    return out


def get_predictions_proba(model, X, a=None, device='cpu'):
    """
    Get predicted probabilities from any model type.
    
    Args:
        model: Trained model (can be sklearn, PyTorch, or custom)
        X: Features (n_samples, n_features)
        a: Sensitive attributes (optional, for some post-processing methods)
        device: Device for PyTorch models
        
    Returns:
        probs: Predicted probabilities (n_samples,)
    """
    import torch
    
    # Try predict_proba method first (sklearn-style or custom)
    if hasattr(model, 'predict_proba'):
        try:
            # Some methods need sensitive attribute
            return model.predict_proba(X, a=a)
        except TypeError:
            # Others don't
            try:
                probs = model.predict_proba(X)
                # Handle sklearn format (n_samples, 2)
                if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] == 2:
                    return probs[:, 1]
                return probs
            except:
                pass
    
    # PyTorch model
    tm = model.model if hasattr(model, 'model') else model
    if isinstance(tm, torch.nn.Module):
        tm.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X).float().to(device)
            logits = tm(X_t)
            probs = torch.sigmoid(logits)
            return probs.cpu().numpy().reshape(-1)
    
    # Fallback: use predict and convert to float
    if hasattr(model, 'predict'):
        preds = model.predict(X)
        return preds.astype(float)
    
    raise ValueError(f"Cannot extract predictions from model type: {type(model)}")


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Group metrics
    performance = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    fairness = ['statistical_parity', 'equalized_odds', 'gcf_ig']
    
    print("\n📊 Performance Metrics:")
    for metric in performance:
        if metric in metrics:
            print(f"  {metric:.<30} {metrics[metric]:.4f}")
    
    print("\n⚖️  Fairness Metrics:")
    for metric in fairness:
        if metric in metrics:
            print(f"  {metric:.<30} {metrics[metric]:.4f}")
    
    print("\n📈 Detailed Fairness:")
    if 'tpr_group0' in metrics:
        print(f"  TPR Group 0:................... {metrics['tpr_group0']:.4f}")
        print(f"  TPR Group 1:................... {metrics['tpr_group1']:.4f}")
        print(f"  FPR Group 0:................... {metrics['fpr_group0']:.4f}")
        print(f"  FPR Group 1:................... {metrics['fpr_group1']:.4f}")
        print(f"  TPR Diff:...................... {metrics['tpr_diff']:.4f}")
        print(f"  FPR Diff:...................... {metrics['fpr_diff']:.4f}")
        print(f"  (Fixed: now uses confusion matrix, not probability averages)")