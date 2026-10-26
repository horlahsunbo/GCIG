"""
Data utilities for GCf-IG experiments.
Handles loading, preprocessing, and creating counterfactual pairs.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional


class FairnessDataset:
    """
    Wrapper for fairness datasets with protected attributes.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        feature_names: list = None,
        protected_attr_name: str = "protected",
        group_names: Dict[int, str] = None
    ):
        """
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - binary {0, 1}
            a: Protected attributes (n_samples,) - binary {0, 1}
            feature_names: List of feature names
            protected_attr_name: Name of protected attribute
            group_names: Mapping from {0, 1} to group names, e.g., {0: 'Female', 1: 'Male'}
        """
        self.X = X
        self.y = y
        self.a = a
        self.feature_names = feature_names
        self.protected_attr_name = protected_attr_name
        self.group_names = group_names or {0: "Group0", 1: "Group1"}
        
        # Validate
        assert len(X) == len(y) == len(a), "X, y, a must have same length"
        assert set(np.unique(y)).issubset({0, 1}), "y must be binary {0, 1}"
        assert set(np.unique(a)).issubset({0, 1}), "a must be binary {0, 1}"
    
    def __len__(self):
        return len(self.X)
    
    def get_group_indices(self, group: int, label: Optional[int] = None):
        """Get indices for a specific group and optionally label."""
        mask = (self.a == group)
        if label is not None:
            mask = mask & (self.y == label)
        return np.where(mask)[0]
    
    def get_baseline(self, group: int, label: int, method: str = 'mean'):
        """Compute baseline for (group, label) pair."""
        indices = self.get_group_indices(group, label)
        if len(indices) == 0:
            # Fallback to overall mean for this label
            indices = np.where(self.y == label)[0]
        
        if method == 'mean':
            return self.X[indices].mean(axis=0)
        elif method == 'median':
            return np.median(self.X[indices], axis=0)
        else:
            raise ValueError(f"Unknown baseline method: {method}")


# FIXED load_german_credit function
# Replace in data_utils.py starting around line 40

def load_german_credit(
    filepath: str,
    target_col: str = "GoodCustomer",
    protected_col: str = "Gender",
    test_size: float = 0.25,
    random_state: int = 42,
    scale: bool = True
) -> Tuple[FairnessDataset, FairnessDataset]:
    """
    Load and preprocess German Credit dataset.
    
    Returns:
        train_data: FairnessDataset for training
        test_data: FairnessDataset for testing
    """
    df = pd.read_csv(filepath)
    
    # Map target: -1 -> 0, 1 -> 1
    y = (df[target_col] == 1).astype(int).values  # Add .values
    
    # Map protected attribute: Female -> 0, Male -> 1
    a = (df[protected_col] == 'Male').astype(int).values  # Add .values
    
    # Get features (exclude target and protected attribute)
    feature_cols = [c for c in df.columns if c not in [target_col, protected_col]]
    
    # Handle categorical column: PurposeOfLoan
    # Option 1: One-hot encode it
    df_features = df[feature_cols].copy()
    if 'PurposeOfLoan' in df_features.columns:
        # One-hot encode PurposeOfLoan
        purpose_dummies = pd.get_dummies(df_features['PurposeOfLoan'], prefix='Purpose')
        df_features = df_features.drop('PurposeOfLoan', axis=1)
        df_features = pd.concat([df_features, purpose_dummies], axis=1)
    
    X = df_features.values.astype(np.float32)
    feature_cols = list(df_features.columns)  # Update feature names
    
    # Train-test split stratified by (y, a)
    stratify_key = y * 2 + a  # Creates 4 groups
    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=random_state, stratify=stratify_key
    )
    
    # Scale features
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Create FairnessDataset objects
    train_data = FairnessDataset(
        X=X_train,
        y=y_train,
        a=a_train,
        feature_names=feature_cols,
        protected_attr_name=protected_col,
        group_names={0: 'Female', 1: 'Male'}
    )
    
    test_data = FairnessDataset(
        X=X_test,
        y=y_test,
        a=a_test,
        feature_names=feature_cols,
        protected_attr_name=protected_col,
        group_names={0: 'Female', 1: 'Male'}
    )
    
    return train_data, test_data


def load_dataset(
    name: str,
    filepath: str,
    target_col: str,
    protected_col: str,
    protected_mapping: Dict[str, int],
    target_mapping: Dict[str, int] = None,
    test_size: float = 0.25,
    random_state: int = 42,
    scale: bool = True
) -> Tuple[FairnessDataset, FairnessDataset]:
    """
    Generic dataset loader - can be extended for other datasets.
    
    Args:
        name: Dataset name (for custom logic if needed)
        filepath: Path to CSV file
        target_col: Name of target column
        protected_col: Name of protected attribute column
        protected_mapping: Map from original values to {0, 1}
        target_mapping: Map from original values to {0, 1} (if None, assumes already binary)
        test_size: Test set proportion
        random_state: Random seed
        scale: Whether to standardize features
        
    Returns:
        train_data, test_data
    """
    df = pd.read_csv(filepath)
    
    # Map target
    if target_mapping:
        y = df[target_col].map(target_mapping).values
    else:
        y = df[target_col].values
    y = y.astype(int)
    
    # Map protected attribute
    a = df[protected_col].map(protected_mapping).values.astype(int)
    
    # Get features
    feature_cols = [c for c in df.columns if c not in [target_col, protected_col]]
    X = df[feature_cols].values.astype(np.float32)
    
    # Train-test split
    stratify_key = y * 2 + a
    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=random_state, stratify=stratify_key
    )
    
    # Scale
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Get group names (reverse mapping)
    group_names = {v: k for k, v in protected_mapping.items()}
    
    train_data = FairnessDataset(
        X=X_train,
        y=y_train,
        a=a_train,
        feature_names=feature_cols,
        protected_attr_name=protected_col,
        group_names=group_names
    )
    
    test_data = FairnessDataset(
        X=X_test,
        y=y_test,
        a=a_test,
        feature_names=feature_cols,
        protected_attr_name=protected_col,
        group_names=group_names
    )
    
    return train_data, test_data
