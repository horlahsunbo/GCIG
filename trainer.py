"""
Training procedures for GCf-IG models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from losses import GCfIGLoss, StatisticalParityLoss, EqualizedOddsLoss
from data_utils import FairnessDataset


class BaselineManager:
    """
    Manages exponential moving average baselines for each (label, group) pair.
    """
    
    def __init__(
        self,
        n_features: int,
        ema_rate: float = 0.1,
        device: str = 'cpu'
    ):
        """
        Args:
            n_features: Dimensionality of features
            ema_rate: EMA coefficient (gamma in algorithm)
            device: 'cpu' or 'cuda'
        """
        self.n_features = n_features
        self.gamma = ema_rate
        self.device = device
        
        # Initialize baselines as zeros
        self.baselines = {
            (0, 0): torch.zeros(n_features, device=device),
            (0, 1): torch.zeros(n_features, device=device),
            (1, 0): torch.zeros(n_features, device=device),
            (1, 1): torch.zeros(n_features, device=device),
        }
        self.initialized = {k: False for k in self.baselines.keys()}
    
    def initialize_from_data(self, data: FairnessDataset):
        """Initialize baselines from dataset means."""
        for label in [0, 1]:
            for group in [0, 1]:
                baseline = data.get_baseline(group, label, method='mean')
                self.baselines[(label, group)] = torch.from_numpy(baseline).float().to(self.device)
                self.initialized[(label, group)] = True
    
    def update_from_batch(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        a: torch.Tensor
    ):
        """
        Update baselines using EMA from batch statistics.
        
        Args:
            X: Features (batch_size, n_features)
            y: Labels (batch_size,)
            a: Protected attributes (batch_size,)
        """
        for label in [0, 1]:
            for group in [0, 1]:
                mask = (y == label) & (a == group)
                if mask.sum() > 0:
                    batch_mean = X[mask].mean(dim=0).detach()
                    
                    if not self.initialized[(label, group)]:
                        self.baselines[(label, group)] = batch_mean
                        self.initialized[(label, group)] = True
                    else:
                        # EMA update
                        old_baseline = self.baselines[(label, group)]
                        self.baselines[(label, group)] = (1 - self.gamma) * old_baseline + self.gamma * batch_mean
    
    def get_baselines(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Return current baselines (detached for loss computation)."""
        return {k: v.detach() for k, v in self.baselines.items()}


class GCfIGTrainer:
    """
    Trainer implementing Algorithm 2: GCf-IG Regularized Training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = 'cpu'
    ):
        """
        Args:
            model: Neural network
            config: Training configuration dict
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Hyperparameters
        self.lr = config.get('learning_rate', 0.001)
        self.n_epochs = config.get('n_epochs', 50)
        self.batch_size = config.get('batch_size', 128)
        
        # Loss weights
        self.lambda_ig = config.get('lambda_ig', 1.0)
        self.lambda_sp = config.get('lambda_sp', 1.0)
        self.lambda_eo = config.get('lambda_eo', 0.0)  # Optional EO instead of SP
        
        # GCf-IG parameters
        self.ig_steps = config.get('ig_steps', 32)
        self.ema_rate = config.get('ema_rate', 0.1)
        self.norm = config.get('norm', 'l2')
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Loss functions
        self.pred_loss_fn = nn.BCEWithLogitsLoss()
        self.gcf_ig_loss_fn = GCfIGLoss(
            n_integration_steps=self.ig_steps,
            norm=self.norm,
            stratify_by_label=True
        )
        self.sp_loss_fn = StatisticalParityLoss()
        self.eo_loss_fn = EqualizedOddsLoss()
        
        # Baseline manager
        self.baseline_manager = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_pred_loss': [],
            'train_gcf_ig_loss': [],
            'train_sp_loss': [],
        }
    
    def train(
        self,
        train_data: FairnessDataset,
        val_data: Optional[FairnessDataset] = None,
        verbose: bool = True
    ):
        """
        Train the model using GCf-IG regularization.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset (optional)
            verbose: Whether to show progress bars
        """
        # Initialize baseline manager
        self.baseline_manager = BaselineManager(
            n_features=train_data.X.shape[1],
            ema_rate=self.ema_rate,
            device=self.device
        )
        self.baseline_manager.initialize_from_data(train_data)
        
        # Create data loader
        train_dataset = TensorDataset(
            torch.from_numpy(train_data.X).float(),
            torch.from_numpy(train_data.y).long(),
            torch.from_numpy(train_data.a).long()
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # Training loop
        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_losses = {
                'total': 0.0,
                'pred': 0.0,
                'gcf_ig': 0.0,
                'sp': 0.0,
            }
            n_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}") if verbose else train_loader
            
            for batch_X, batch_y, batch_a in pbar:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_a = batch_a.to(self.device)
                
                # (1) Update baselines from batch statistics
                self.baseline_manager.update_from_batch(batch_X, batch_y, batch_a)
                baselines = self.baseline_manager.get_baselines()
                
                # (2) Forward pass
                logits = self.model(batch_X)
                
                # (3) Prediction loss
                pred_loss = self.pred_loss_fn(logits, batch_y.float())
                
                # (4) GCf-IG loss
                gcf_ig_loss = self.gcf_ig_loss_fn(
                    self.model,
                    batch_X,
                    batch_y,
                    batch_a,
                    baselines
                )
                
                # (5) Fairness loss (SP or EO)
                probs = torch.sigmoid(logits)
                if self.lambda_eo > 0:
                    fairness_loss = self.eo_loss_fn(probs, batch_y, batch_a)
                else:
                    fairness_loss = self.sp_loss_fn(probs, batch_a)
                
                # (6) Total loss
                total_loss = (
                    pred_loss +
                    self.lambda_ig * gcf_ig_loss +
                    self.lambda_sp * fairness_loss
                )
                
                # (7) Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Track losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['pred'] += pred_loss.item()
                epoch_losses['gcf_ig'] += gcf_ig_loss.item()
                epoch_losses['sp'] += fairness_loss.item()
                n_batches += 1
                
                if verbose:
                    pbar.set_postfix({
                        'loss': f"{total_loss.item():.4f}",
                        'gcf_ig': f"{gcf_ig_loss.item():.4f}",
                        'sp': f"{fairness_loss.item():.4f}"
                    })
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= n_batches
            
            self.history['train_loss'].append(epoch_losses['total'])
            self.history['train_pred_loss'].append(epoch_losses['pred'])
            self.history['train_gcf_ig_loss'].append(epoch_losses['gcf_ig'])
            self.history['train_sp_loss'].append(epoch_losses['sp'])
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Loss={epoch_losses['total']:.4f}, "
                      f"GCf-IG={epoch_losses['gcf_ig']:.4f}, SP={epoch_losses['sp']:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions (probabilities).
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            probs: Predicted probabilities (n_samples,)
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy()
    
    def predict_labels(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Get binary predictions.
        
        Args:
            X: Features (n_samples, n_features)
            threshold: Classification threshold
            
        Returns:
            labels: Predicted labels (n_samples,)
        """
        probs = self.predict(X)
        return (probs >= threshold).astype(int)
