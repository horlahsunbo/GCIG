# ===== FILE: baselines.py =====
"""
Baseline fairness methods for comparison.

Implements:
- Hardt et al. (2016): Post-processing for Equalized Odds
- Agarwal et al. (2018): Reductions approach
- Lagrangian method (Cotter et al. 2019)
- Adversarial Debiasing
- Feldman et al. (2015): Disparate Impact Remover (pre-processing)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds as EO_Constraint
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    print("Warning: fairlearn not available. Install with: pip install fairlearn")

from models import MLPClassifier




# Module-level wrapper for Hardt post-processing compatibility
class _HardtSkWrapper:
    """Sklearn-compatible wrapper for base model used by Hardt method."""
    def __init__(self, model):
        self.model = model
    
    def get_params(self, deep=True):
        """Required by sklearn clone."""
        return {'model': self.model}
    
    def set_params(self, **params):
        """Required by sklearn clone."""
        if 'model' in params:
            self.model = params['model']
        return self
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        """Predict class labels."""
        import torch
        import numpy as np
        
        if hasattr(self.model, 'predict'):
            # Sklearn-style model
            return self.model.predict(X)
        else:
            # PyTorch model
            probs = self.predict_proba(X)
            # probs is (n, 2) array, take argmax
            if isinstance(probs, np.ndarray) and probs.ndim == 2:
                return np.argmax(probs, axis=1)
            # probs is (n,) array of class 1 probabilities
            return (probs >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """Predict probabilities for ThresholdOptimizer."""
        import torch
        import numpy as np
        
        if hasattr(self.model, 'predict_proba'):
            # Sklearn-style model
            proba = self.model.predict_proba(X)
            if isinstance(proba, np.ndarray) and proba.ndim == 1:
                proba_neg = 1 - proba
                return np.column_stack([proba_neg, proba])
            return proba
        else:
            # PyTorch model (MLPClassifier)
            self.model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X)
                # Check if model is on GPU
                if next(self.model.parameters()).is_cuda:
                    X_t = X_t.cuda()
                
                # Forward pass
                logits = self.model(X_t).squeeze()
                probs_pos = torch.sigmoid(logits)
            
            # Return (n, 2) array: [prob_class_0, prob_class_1]
            probs_pos = probs_pos.cpu().numpy()
            probs_neg = 1 - probs_pos
            return np.column_stack([probs_neg, probs_pos])

class BaselineMethod:
    """Base class for fairness methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        
    def fit(self, X_train, y_train, a_train):
        raise NotImplementedError
        
    def predict(self, X):
        raise NotImplementedError
    
    def predict_proba(self, X):
        raise NotImplementedError


@dataclass
class _DIRQuantiles:
    """Quantile information for Feldman-style Disparate Impact Remover.

    We implement a lightweight, dependency-free variant of the Feldman
    "repair" idea: for each continuous feature, map each group's values to a
    common (barycenter) distribution via quantile matching.
    """
    feature_indices: List[int]
    # For each feature index j -> dict: group -> sorted values
    sorted_by_group: Dict[int, Dict[int, np.ndarray]]
    # For each feature index j -> barycenter quantile values on q_grid
    barycenter: Dict[int, np.ndarray]
    # Quantile grid used for interpolation
    q_grid: np.ndarray


class DisparateImpactRemover:
    """Feldman et al.-style disparate impact remover (pre-processing).

    Notes:
    - Practical approximation for tabular data.
    - Operates feature-wise on continuous-ish columns.
    - Leaves {0,1} one-hot/binary columns unchanged.
    """

    def __init__(self, repair_level: float = 1.0, n_quantiles: int = 200):
        assert 0.0 <= repair_level <= 1.0
        self.repair_level = repair_level
        self.n_quantiles = n_quantiles
        self.stats: Optional[_DIRQuantiles] = None

    def _select_continuous_features(self, X: np.ndarray) -> List[int]:
        idxs = []
        for j in range(X.shape[1]):
            col = X[:, j]
            uniq = np.unique(col)
            if len(uniq) <= 2 and set(uniq).issubset({0.0, 1.0}):
                continue
            idxs.append(j)
        return idxs

    def fit(self, X: np.ndarray, a: np.ndarray):
        groups = sorted(list(set(a.tolist())))
        q_grid = np.linspace(0.0, 1.0, self.n_quantiles)

        feature_indices = self._select_continuous_features(X)
        sorted_by_group: Dict[int, Dict[int, np.ndarray]] = {}
        barycenter: Dict[int, np.ndarray] = {}

        for j in feature_indices:
            sorted_by_group[j] = {}
            q_values = []
            for g in groups:
                vals = np.asarray(X[a == g, j], dtype=float)
                if vals.size == 0:
                    vals = np.asarray(X[:, j], dtype=float)
                vals_sorted = np.sort(vals)
                sorted_by_group[j][g] = vals_sorted

                n = len(vals_sorted)
                pos = q_grid * (n - 1)
                lo = np.floor(pos).astype(int)
                hi = np.ceil(pos).astype(int)
                w = pos - lo
                qv = (1 - w) * vals_sorted[lo] + w * vals_sorted[hi]
                q_values.append(qv)

            barycenter[j] = np.mean(np.stack(q_values, axis=0), axis=0)

        self.stats = _DIRQuantiles(
            feature_indices=feature_indices,
            sorted_by_group=sorted_by_group,
            barycenter=barycenter,
            q_grid=q_grid,
        )
        return self

    def transform(self, X: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.stats is None or self.repair_level == 0.0:
            return X

        X_new = X.astype(float).copy()
        groups = sorted(list(set(a.tolist())))
        q_grid = self.stats.q_grid

        for j in self.stats.feature_indices:
            q_bar = self.stats.barycenter[j]
            for g in groups:
                idx = np.where(a == g)[0]
                if idx.size == 0:
                    continue
                vals = X_new[idx, j]
                vals_sorted = self.stats.sorted_by_group[j][g]
                n = len(vals_sorted)
                if n <= 1:
                    continue

                ranks = np.searchsorted(vals_sorted, vals, side='left')
                u = ranks / (n - 1)
                u = np.clip(u, 0.0, 1.0)

                repaired = np.interp(u, q_grid, q_bar)
                X_new[idx, j] = (1 - self.repair_level) * vals + self.repair_level * repaired

        return X_new


class DIRPreprocessMLP(BaselineMethod):
    """Pre-processing baseline: Disparate Impact Remover + unconstrained MLP."""

    def __init__(self, input_dim: int, config: dict, device='cpu', repair_level: float = 1.0):
        super().__init__("Disparate Impact Remover + MLP")
        self.device = device
        self.config = config
        self.repair_level = repair_level
        self.dir = DisparateImpactRemover(repair_level=repair_level)

        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [64, 32]),
            dropout=config.get('dropout', 0.1)
        ).to(device)

    def fit(self, X_train, y_train, a_train):
        self.dir.fit(X_train, a_train)
        Xr = self.dir.transform(X_train, a_train)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get('learning_rate', 0.001))
        loss_fn = nn.BCEWithLogitsLoss()

        X_t = torch.FloatTensor(Xr).to(self.device)
        y_t = torch.FloatTensor(y_train).to(self.device)

        self.model.train()
        for _ in range(self.config.get('n_epochs', 50)):
            optimizer.zero_grad()
            logits = self.model(X_t).squeeze()
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()
        return self

    def _transform_infer(self, X, a=None):
        if a is None:
            return X
        return self.dir.transform(X, a)

    def predict_proba(self, X, a=None):
        Xr = self._transform_infer(X, a)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(Xr).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy().reshape(-1)

    def predict(self, X, a=None):
        p = self.predict_proba(X, a=a)
        return (p >= 0.5).astype(int)


class UnconstrainedMLP(BaselineMethod):
    """Standard MLP without fairness constraints (baseline)"""
    
    def __init__(self, input_dim: int, config: dict, device='cpu'):
        super().__init__("Unconstrained MLP")
        self.device = device
        self.config = config
        
        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [64, 32]),
            dropout=config.get('dropout', 0.1)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001)
        )
        self.criterion = nn.BCEWithLogitsLoss()
        
    def fit(self, X_train, y_train, a_train, epochs=50, batch_size=64, verbose=False):
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(X_batch).squeeze()
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
                
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        return preds.cpu().numpy().astype(int).squeeze()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy().squeeze()


class HardtMethod(BaselineMethod):
    """
    Hardt et al. (2016): Post-processing for Equalized Odds
    Adjusts thresholds per group to satisfy EO constraints.
    """
    
    def __init__(self, input_dim=None, config=None, base_model=None, device='cpu'):
        super().__init__("Hardt (Post-processing)")
        self.device = device
        self.config = config or {}
        
        # Use neural network base model
        if base_model is not None:
            self.base_model = base_model
        elif input_dim is not None:
            # Create MLP base model
            self.base_model = MLPClassifier(
                input_dim=input_dim,
                hidden_dims=self.config.get('hidden_dims', [64, 32]),
                dropout=self.config.get('dropout', 0.1)
            ).to(device)
        else:
            self.base_model = None
        
        self.postprocessor = None
        # For explanation metrics (e.g., GCf-IG), we attribute to the underlying neural network
        self.explanation_model = self.base_model
        
    
    def fit(self, X_train, y_train, a_train, X_val=None, y_val=None, a_val=None, epochs=50, verbose=False):
        """Fit Hardt post-processing with MLP base model."""
        if not FAIRLEARN_AVAILABLE:
            raise ImportError("fairlearn required for Hardt method")
        
        # Train MLP base model if needed
        if self.base_model is not None and hasattr(self.base_model, 'parameters'):
            optimizer = torch.optim.Adam(
                self.base_model.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
            loss_fn = nn.BCEWithLogitsLoss()
            
            X_t = torch.FloatTensor(X_train).to(self.device)
            y_t = torch.FloatTensor(y_train).to(self.device)
            
            self.base_model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                logits = self.base_model(X_t).squeeze()
                loss = loss_fn(logits, y_t)
                loss.backward()
                optimizer.step()
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"      Hardt base - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Wrap base model for fairlearn
        base_for_post = _HardtSkWrapper(self.base_model)
        self.explanation_model = self.base_model
        
        # Use validation or training set
        X_post = X_val if X_val is not None else X_train
        y_post = y_val if y_val is not None else y_train
        a_post = a_val if a_val is not None else a_train
        
        # Fit threshold optimizer
        self.postprocessor = ThresholdOptimizer(
            estimator=base_for_post,
            constraints="equalized_odds",
            predict_method="predict_proba"
        )
        
        self.postprocessor.fit(X_post, y_post, sensitive_features=a_post)
        return self

    def predict(self, X, a=None):
        if a is not None and self.postprocessor is not None:
            return self.postprocessor.predict(X, sensitive_features=a)
        # Fallback: threshold raw probabilities
        p = self.predict_proba(X, a=a)
        return (p >= 0.5).astype(int)
    
    def predict_proba(self, X, a=None):
        if hasattr(self.base_model, 'predict_proba'):
            # sklearn-style or wrapped baseline
            p = self.base_model.predict_proba(X)
            if isinstance(p, np.ndarray) and p.ndim == 2:
                return p[:, 1]
            return np.asarray(p).reshape(-1)
        else:
            # PyTorch model
            self.base_model.eval()
            with torch.no_grad():
                X_t = torch.FloatTensor(X).to(self.device)
                logits = self.base_model(X_t)
                probs = torch.sigmoid(logits)
            return probs.cpu().numpy().reshape(-1)


class AgarwalMethod(BaselineMethod):
    """
    Agarwal et al. (2018): Reductions approach
    Reduces fair classification to cost-sensitive classification.
    """
    
    def __init__(self, input_dim=None, config=None, constraint='equalized_odds', device='cpu'):
        super().__init__("Agarwal (Reductions)")
        self.constraint = constraint
        self.device = device
        self.config = config or {}
        self.input_dim = input_dim
        self.mitigator = None
        
        # Create base MLP for explanation
        if input_dim is not None:
            self.explanation_model = MLPClassifier(
                input_dim=input_dim,
                hidden_dims=self.config.get('hidden_dims', [64, 32]),
                dropout=self.config.get('dropout', 0.1)
            ).to(device)
        else:
            self.explanation_model = None
        
    def fit(self, X_train, y_train, a_train, epochs=50, verbose=False):
        if not FAIRLEARN_AVAILABLE:
            raise ImportError("fairlearn required for Agarwal method")
        
        # First, train the explanation model (MLP) for GCf-IG computation
        if self.explanation_model is not None:
            optimizer = torch.optim.Adam(
                self.explanation_model.parameters(),
                lr=self.config.get('learning_rate', 0.001)
            )
            loss_fn = nn.BCEWithLogitsLoss()
            
            X_t = torch.FloatTensor(X_train).to(self.device)
            y_t = torch.FloatTensor(y_train).to(self.device)
            
            self.explanation_model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                logits = self.explanation_model(X_t).squeeze()
                loss = loss_fn(logits, y_t)
                loss.backward()
                optimizer.step()
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"      Agarwal base training - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Choose constraint
        if self.constraint == 'equalized_odds':
            constraint_obj = EO_Constraint()
        elif self.constraint == 'demographic_parity':
            constraint_obj = DemographicParity()
        else:
            raise ValueError(f"Unknown constraint: {self.constraint}")
        
        # Use LogisticRegression for the reduction (fast, required by fairlearn)
        base_estimator = LogisticRegression(max_iter=1000, solver='saga')
        
        # Exponential gradient reduction
        self.mitigator = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint_obj,
            eps=0.01,  # Fairness tolerance
            max_iter=50  # Use fixed iterations for reductions
        )
        
        self.mitigator.fit(X_train, y_train, sensitive_features=a_train)
    
    def predict(self, X, a=None):
        if self.mitigator is None:
            raise ValueError("Model not fitted")
        if a is None:
            return self.mitigator.predict(X)
        return self.mitigator.predict(X, sensitive_features=a)
    
    def predict_proba(self, X, a=None):
        if self.mitigator is None:
            raise ValueError("Model not fitted")
        # fairlearn mitigator may not expose predict_proba reliably; approximate via predict
        preds = self.predict(X, a=a)
        return preds.astype(float)


class LagrangianMethod(BaselineMethod):
    """
    Lagrangian method for equalized odds constraints.
    Similar to Cotter et al. (2019).
    """
    
    def __init__(self, input_dim: int, config: dict, device='cpu'):
        super().__init__("Lagrangian (In-processing)")
        self.device = device
        self.config = config
        
        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [64, 32]),
            dropout=config.get('dropout', 0.1)
        ).to(device)
        
        # Lagrange multipliers for EO constraints
        self.lambda_tpr = torch.tensor(0.0, device=device, requires_grad=False)
        self.lambda_fpr = torch.tensor(0.0, device=device, requires_grad=False)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        self.eta_lambda = config.get('eta_lambda', 0.01)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def compute_eo_penalty(self, y_true, y_pred, a):
        """Compute EO penalty for current batch."""
        # Convert to numpy for easier group computations
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        a_np = a.cpu().numpy()
        
        # Compute TPR and FPR for each group
        tpr = {}
        fpr = {}
        
        for g in [0, 1]:
            mask_g = (a_np == g)
            
            # TPR
            mask_tp = mask_g & (y_true_np == 1)
            if mask_tp.sum() > 0:
                tpr[g] = y_pred_np[mask_tp].mean()
            else:
                tpr[g] = 0.0
            
            # FPR
            mask_fp = mask_g & (y_true_np == 0)
            if mask_fp.sum() > 0:
                fpr[g] = y_pred_np[mask_fp].mean()
            else:
                fpr[g] = 0.0
        
        # Compute squared differences
        tpr_diff = (tpr[0] - tpr[1]) ** 2
        fpr_diff = (fpr[0] - fpr[1]) ** 2
        
        return tpr_diff, fpr_diff
    
    def fit(self, X_train, y_train, a_train, epochs=50, batch_size=64, verbose=False):
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(a_train)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for X_batch, y_batch, a_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                a_batch = a_batch.to(self.device)
                
                # Forward pass
                self.model.train()
                logits = self.model(X_batch).squeeze()
                probs = torch.sigmoid(logits)
                
                # Prediction loss
                pred_loss = self.criterion(logits, y_batch)
                
                # EO penalty
                tpr_diff, fpr_diff = self.compute_eo_penalty(y_batch, probs, a_batch)
                
                # Total loss with Lagrangian
                eo_loss = self.lambda_tpr * tpr_diff + self.lambda_fpr * fpr_diff
                loss = pred_loss + eo_loss
                
                # Update model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update Lagrange multipliers (gradient ascent)
                self.lambda_tpr = torch.clamp(
                    self.lambda_tpr + self.eta_lambda * tpr_diff, min=0.0
                )
                self.lambda_fpr = torch.clamp(
                    self.lambda_fpr + self.eta_lambda * fpr_diff, min=0.0
                )
                
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, "
                      f"Lambda TPR: {self.lambda_tpr:.4f}, Lambda FPR: {self.lambda_fpr:.4f}")
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        return preds.cpu().numpy().astype(int).squeeze()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy().squeeze()


class AdversarialDebiasing(BaselineMethod):
    """
    Adversarial debiasing method.
    Trains classifier to be predictive while minimizing protected attribute predictability.
    """
    
    def __init__(self, input_dim: int, config: dict, device='cpu'):
        super().__init__("Adversarial Debiasing (In-processing)")
        self.device = device
        self.config = config
        
        # Classifier
        self.model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [64, 32]),
            dropout=config.get('dropout', 0.1)
        ).to(device)
        
        # Adversary predicts protected attribute from classifier output
        self.adversary = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(device)
        
        self.clf_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        self.adv_optimizer = torch.optim.Adam(
            self.adversary.parameters(),
            lr=config.get('adv_learning_rate', 0.001)
        )
        
        self.clf_criterion = nn.BCEWithLogitsLoss()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        
        self.beta = config.get('beta_adv', 1.0)  # Adversarial strength
        
    def fit(self, X_train, y_train, a_train, epochs=50, batch_size=64, verbose=False):
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(a_train)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_clf_loss = 0
            total_adv_loss = 0
            
            for X_batch, y_batch, a_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                a_batch = a_batch.to(self.device)
                
                # 1. Train adversary
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(X_batch).squeeze()
                
                adv_pred = self.adversary(logits.unsqueeze(1)).squeeze()
                adv_loss = self.adv_criterion(adv_pred, a_batch)
                
                self.adv_optimizer.zero_grad()
                adv_loss.backward()
                self.adv_optimizer.step()
                
                # 2. Train classifier to minimize prediction loss and maximize adversary loss
                self.model.train()
                logits = self.model(X_batch).squeeze()
                probs = torch.sigmoid(logits)
                
                pred_loss = self.clf_criterion(logits, y_batch)
                
                # Adversary prediction (with gradient)
                adv_pred = self.adversary(logits.unsqueeze(1)).squeeze()
                adv_loss = self.adv_criterion(adv_pred, a_batch)
                
                # Total classifier loss
                clf_loss = pred_loss - self.beta * adv_loss
                
                self.clf_optimizer.zero_grad()
                clf_loss.backward()
                self.clf_optimizer.step()
                
                total_clf_loss += clf_loss.item()
                total_adv_loss += adv_loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Clf Loss: {total_clf_loss/len(dataloader):.4f}, "
                      f"Adv Loss: {total_adv_loss/len(dataloader):.4f}")
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
        return preds.cpu().numpy().astype(int).squeeze()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_t)
            probs = torch.sigmoid(logits)
        return probs.cpu().numpy().squeeze()


def train_baseline_method(
    method_name: str,
    train_data,
    test_data,
    config: dict,
    device='cpu',
    verbose=False
) -> Tuple[BaselineMethod, Dict]:
    """
    Train a baseline fairness method.
    
    Args:
        method_name: One of ['unconstrained', 'dir', 'hardt', 'lagrangian', 'adversarial', 'gcf_ig']
        train_data: FairnessDataset
        test_data: FairnessDataset
        config: Configuration dict
        device: 'cpu' or 'cuda'
        verbose: Print training progress
        
    Returns:
        method: Trained method
        results: Dict with train/test metrics
    """
    from metrics import evaluate_model, build_label_group_baselines
    
    input_dim = train_data.X.shape[1]
    
    # Build baselines once from training data for evaluation-time GCf-IG metric
    baselines = build_label_group_baselines(train_data.X, train_data.y, train_data.a, method="mean")

    # Create method
    if method_name == 'unconstrained':
        method = UnconstrainedMLP(input_dim, config, device)
        method.fit(train_data.X, train_data.y, train_data.a, 
                  epochs=config.get('n_epochs', 50), verbose=verbose)
    
    elif method_name in ('dir', 'disparate_impact_remover'):
        method = DIRPreprocessMLP(input_dim, config, device, repair_level=config.get('dir_repair_level', 1.0))
        method.fit(train_data.X, train_data.y, train_data.a)

    elif method_name == 'hardt':
        # Hardt with MLP base model
        method = HardtMethod(input_dim=input_dim, config=config, device=device)
        method.fit(train_data.X, train_data.y, train_data.a,
                  epochs=config.get('n_epochs', 50), verbose=verbose)
    
    elif method_name == 'lagrangian':
        method = LagrangianMethod(input_dim, config, device)
        method.fit(train_data.X, train_data.y, train_data.a,
                  epochs=config.get('n_epochs', 50), verbose=verbose)
    
    elif method_name == 'adversarial':
        method = AdversarialDebiasing(input_dim, config, device)
        method.fit(train_data.X, train_data.y, train_data.a,
                  epochs=config.get('n_epochs', 50), verbose=verbose)
        
    elif method_name == 'agarwal':
        # Agarwal with MLP explanation model
        method = AgarwalMethod(input_dim=input_dim, config=config, 
                              constraint='equalized_odds', device=device)
        method.fit(train_data.X, train_data.y, train_data.a,
                  epochs=config.get('n_epochs', 50), verbose=verbose)
    
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    # Evaluate (include GCf-IG for neural network methods)
    # All methods except sklearn-only DIR can compute GCf-IG now
    compute_gcf = method_name in ['unconstrained', 'dir', 'lagrangian', 'adversarial', 'hardt', 'agarwal', 'gcf_ig']
    train_metrics = evaluate_model(method, train_data, baselines=baselines, device=device, compute_gcf_ig=compute_gcf)
    test_metrics = evaluate_model(method, test_data, baselines=baselines, device=device, compute_gcf_ig=compute_gcf)
    
    results = {
        'train': train_metrics,
        'test': test_metrics
    }
    
    return method, results
