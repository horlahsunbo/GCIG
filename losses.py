# ===== FILE: losses.py =====
"""
Loss functions for GCf-IG fairness regularization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class GCfIGLoss(nn.Module):
    """
    GCf-IG Loss for binary protected attributes and binary labels.
    
    Computes integrated gradients from group-conditional baselines
    and penalizes differences in attribution patterns.
    """
    
    def __init__(
        self,
        n_integration_steps: int = 32,
        norm: str = 'l2',  # 'l1' or 'l2' for normalization
        epsilon: float = 1e-8,
        stratify_by_label: bool = True,  # For EO-alignment
        label_weights: Optional[Dict[int, float]] = None,
        create_graph: bool = True
    ):
        """
        Args:
            n_integration_steps: Number of steps for Integrated Gradients (T)
            norm: Norm for IG normalization ('l1' or 'l2')
            epsilon: Small constant for numerical stability
            stratify_by_label: If True, compute IG loss per label (EO-aligned)
            label_weights: Weights for each label {0: w0, 1: w1}
        """
        super().__init__()
        self.T = n_integration_steps
        self.norm = norm
        self.epsilon = epsilon
        self.stratify_by_label = stratify_by_label
        self.label_weights = label_weights or {0: 0.5, 1: 0.5}
        self.create_graph = create_graph
    
    def compute_ig_from_baselines(
        self,
        model: nn.Module,
        x: torch.Tensor,
        baselines: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients from multiple baselines simultaneously.
        
        Args:
            model: Neural network (outputs logits)
            x: Single input (1, n_features)
            baselines: Multiple baselines (n_baselines, n_features)
            
        Returns:
            ig: Integrated gradients (n_baselines, n_features)
        """
        n_baselines = baselines.shape[0]
        
        # Expand input to match baselines
        x_exp = x.expand(n_baselines, -1)
        
        # Accumulate gradients
        grad_sum = torch.zeros_like(baselines)
        
        # Compute integrated gradients
        for t in range(1, self.T + 1):
            alpha = t / self.T
            
            # Interpolate
            x_t = baselines + alpha * (x_exp - baselines)
            x_t.requires_grad_(True)
            
            # Forward pass
            logits = model(x_t).squeeze()
            logits_sum = logits.sum()
            
            # Compute gradients
            g_t = torch.autograd.grad(logits_sum, x_t, create_graph=self.create_graph)[0]
            grad_sum += g_t
        
        # Average gradients and scale
        avg_grad = grad_sum / self.T
        ig = (x_exp - baselines) * avg_grad
        
        return ig
    
    def normalize_ig(self, ig: torch.Tensor) -> torch.Tensor:
        """Normalize IG attributions."""
        if self.norm == 'l1':
            norm_val = torch.norm(ig, p=1, dim=-1, keepdim=True)
        else:  # l2
            norm_val = torch.norm(ig, p=2, dim=-1, keepdim=True)
        
        return ig / (norm_val + self.epsilon)
    
    def forward(
        self,
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor,
        baselines: Dict[Tuple[int, int], torch.Tensor],
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute GCf-IG loss.
        
        Args:
            model: Neural network (outputs logits)
            X: Input batch (batch_size, n_features)
            y: Labels (batch_size,)
            a: Protected attributes (batch_size,)
            baselines: Dict mapping (label, group) -> baseline vector
            sample_weight: Optional sample weights
            
        Returns:
            loss: GCf-IG loss scalar
        """
        batch_size = X.shape[0]
        device = X.device
        
        # Group identifiers (assume binary for now)
        groups = [0, 1]
        labels = [0, 1]
        
        # Initialize loss
        total_loss = 0.0
        n_valid_labels = 0
        
        # Stratify by label for EO alignment
        if self.stratify_by_label:
            for label in labels:
                label_mask = (y == label)
                
                if label_mask.sum() == 0:
                    continue
                
                # Compute loss for this label
                label_loss = 0.0
                label_count = 0
                
                X_label = X[label_mask]
                a_label = a[label_mask]
                
                for i in range(X_label.shape[0]):
                    x_i = X_label[i:i+1]  # (1, n_features)
                    
                    # Get baselines for this label and all groups
                    label_baselines = []
                    for g in groups:
                        key = (label, g)
                        if key in baselines:
                            label_baselines.append(baselines[key])
                        else:
                            # Fallback to mean of this label
                            label_baselines.append(X_label.mean(dim=0))
                    
                    label_baselines = torch.stack(label_baselines)  # (2, n_features)
                    
                    # Compute IG from each baseline
                    ig = self.compute_ig_from_baselines(model, x_i, label_baselines)
                    
                    # Normalize
                    ig_norm = self.normalize_ig(ig)
                    
                    # Compute variance across baselines (groups)
                    # For 2 groups: Var = 1/4 * ||IG1 - IG2||^2
                    ig_diff = ig_norm[0] - ig_norm[1]
                    variance = 0.25 * torch.sum(ig_diff ** 2)
                    
                    label_loss += torch.sqrt(variance + self.epsilon)
                    label_count += 1
                
                if label_count > 0:
                    label_loss = label_loss / label_count
                    
                    # Weight by label importance
                    weight = self.label_weights.get(label, 0.5)
                    total_loss += weight * label_loss
                    n_valid_labels += 1
        
        else:
            # Non-stratified version (not recommended)
            for i in range(batch_size):
                x_i = X[i:i+1]
                label = y[i].item()
                
                # Get baselines for this sample's label
                label_baselines = []
                for g in groups:
                    key = (label, g)
                    if key in baselines:
                        label_baselines.append(baselines[key])
                    else:
                        label_baselines.append(X.mean(dim=0))
                
                label_baselines = torch.stack(label_baselines)
                
                # Compute IG
                ig = self.compute_ig_from_baselines(model, x_i, label_baselines)
                ig_norm = self.normalize_ig(ig)
                
                # Variance
                ig_diff = ig_norm[0] - ig_norm[1]
                variance = 0.25 * torch.sum(ig_diff ** 2)
                
                total_loss += torch.sqrt(variance + self.epsilon)
        
            total_loss = total_loss / batch_size
        
        return total_loss


class EqualizedOddsLoss(nn.Module):
    """
    Equalized Odds loss for binary protected attribute.
    Penalizes TPR and FPR differences between groups.
    """
    
    def __init__(self, squared: bool = True, epsilon: float = 1e-8):
        super().__init__()
        self.squared = squared
        self.epsilon = epsilon
    
    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        a: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute EO loss.
        
        Args:
            y_true: True labels (batch_size,)
            y_pred: Predicted probabilities (batch_size,)
            a: Protected attributes (batch_size,)
            sample_weight: Optional weights
            
        Returns:
            eo_loss: Equalized odds loss
        """
        if y_pred.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
            
            y_pred = y_pred.float()
        groups = [0, 1]
        
        # Compute TPR and FPR for each group
        tpr = {}
        fpr = {}
        
        for g in groups:
            mask_g = (a == g)
            
            # True positives
            mask_tp = mask_g & (y_true == 1)
            if mask_tp.sum() > 0:
                tpr[g] = y_pred[mask_tp].mean()
            else:
                tpr[g] = torch.tensor(0.0, device=y_true.device)
            
            # False positives
            mask_fp = mask_g & (y_true == 0)
            if mask_fp.sum() > 0:
                fpr[g] = y_pred[mask_fp].mean()
            else:
                fpr[g] = torch.tensor(0.0, device=y_true.device)
        
        # Compute differences
        tpr_diff = torch.abs(tpr[0] - tpr[1])
        fpr_diff = torch.abs(fpr[0] - fpr[1])
        
        if self.squared:
            eo_loss = tpr_diff ** 2 + fpr_diff ** 2
        else:
            eo_loss = tpr_diff + fpr_diff
        
        return eo_loss


class StatisticalParityLoss(nn.Module):
    """
    Statistical Parity loss.
    Penalizes difference in positive prediction rates.
    """
    
    def __init__(self, squared: bool = True):
        super().__init__()
        self.squared = squared
    
    def forward(
        self,
        y_pred: torch.Tensor,
        a: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SP loss.
        
        Args:
            y_pred: Predicted probabilities (batch_size,)
            a: Protected attributes (batch_size,)
            sample_weight: Optional weights
            
        Returns:
            sp_loss: Statistical parity loss
        """
        groups = [0, 1]
        
        # Compute positive rates
        pr = {}
        for g in groups:
            mask_g = (a == g)
            if mask_g.sum() > 0:
                pr[g] = y_pred[mask_g].mean()
            else:
                pr[g] = torch.tensor(0.0, device=y_pred.device)
        
        pr_diff = torch.abs(pr[0] - pr[1])
        
        if self.squared:
            return pr_diff ** 2
        else:
            return pr_diff
