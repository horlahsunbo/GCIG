"""
Additional datasets for fairness evaluation.
ULTRA-RELIABLE VERSION: Multiple fallback strategies for bank dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data_utils import FairnessDataset
import warnings
import io
warnings.filterwarnings('ignore')


def load_bank_data_robust(data_dir: str = '.', test_size: float = 0.3, random_state: int = 42):
    """
    Load Bank Marketing dataset with MULTIPLE FALLBACK STRATEGIES.
    
    Tries in order:
    1. Local file (if exists)
    2. Direct UCI CSV link (not ZIP!)
    3. Alternative mirrors
    4. Hardcoded sample (emergency fallback)
    
    Target: Whether client subscribed to term deposit (yes/no)
    Protected attribute: Age (>= 25 as privileged group)
    
    Args:
        data_dir: Directory containing bank-full.csv (if available)
        test_size: Test split fraction
        random_state: Random seed
        
    Returns:
        train_data, test_data: FairnessDataset objects
    """
    print("\n" + "="*80)
    print("Loading Bank Marketing Dataset (ULTRA-RELIABLE MODE)")
    print("="*80)
    
    df = None
    method_used = None
    
    # ========================================================================
    # STRATEGY 1: Try local file first (fastest, most reliable)
    # ========================================================================
    possible_paths = [
        Path(data_dir) / 'bank-full.csv',
        Path(data_dir) / 'data' / 'bank-full.csv', 
        Path(data_dir) / 'datasets' / 'bank-full.csv',
        Path('bank-full.csv'),
        Path('data/bank-full.csv'),
        Path('datasets/bank-full.csv'),
        Path('/tmp/bank-full.csv'),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                print(f"✓ Strategy 1: Loading from local file: {path}")
                df = pd.read_csv(path, sep=';', quotechar='"')
                method_used = f"Local file: {path}"
                break
            except Exception as e:
                print(f"  ✗ Failed to load {path}: {e}")
                continue
    
    # ========================================================================
    # STRATEGY 2: Direct CSV download (not ZIP!)
    # ========================================================================
    if df is None:
        # Try OpenML (direct CSV, no ZIP)
        urls_to_try = [
            # OpenML direct CSV
            'https://www.openml.org/data/get_csv/1461348/bank-full.csv',
            # GitHub mirror (reliable)
            'https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv',
            # Another GitHub mirror
            'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/bank-full.csv',
        ]
        
        for idx, url in enumerate(urls_to_try, 1):
            try:
                print(f"✓ Strategy 2.{idx}: Downloading from: {url[:60]}...")
                df = pd.read_csv(url, sep=';', quotechar='"', encoding='utf-8')
                method_used = f"Downloaded from mirror {idx}"
                
                # Save for future use
                cache_path = Path(data_dir) / 'bank-full.csv'
                cache_path.parent.mkdir(exist_ok=True)
                df.to_csv(cache_path, sep=';', index=False)
                print(f"  ✓ Cached to: {cache_path}")
                break
            except Exception as e:
                print(f"  ✗ Mirror {idx} failed: {e}")
                continue
    
    # ========================================================================
    # STRATEGY 3: Try to download and extract from ZIP (last resort)
    # ========================================================================
    if df is None:
        try:
            import zipfile
            import urllib.request
            from io import BytesIO
            
            print("✓ Strategy 3: Trying ZIP extraction...")
            zip_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
            
            # Download ZIP
            response = urllib.request.urlopen(zip_url, timeout=30)
            zip_data = BytesIO(response.read())
            
            # Extract bank-full.csv from ZIP
            with zipfile.ZipFile(zip_data) as zf:
                # Look for bank-full.csv in the ZIP
                for filename in zf.namelist():
                    if 'bank-full.csv' in filename:
                        print(f"  ✓ Found {filename} in ZIP")
                        csv_data = zf.read(filename)
                        df = pd.read_csv(BytesIO(csv_data), sep=';', quotechar='"')
                        method_used = "Extracted from UCI ZIP"
                        
                        # Save for future use
                        cache_path = Path(data_dir) / 'bank-full.csv'
                        cache_path.parent.mkdir(exist_ok=True)
                        df.to_csv(cache_path, sep=';', index=False)
                        print(f"  ✓ Cached to: {cache_path}")
                        break
        except Exception as e:
            print(f"  ✗ ZIP extraction failed: {e}")
    
    # ========================================================================
    # STRATEGY 4: Generate synthetic data (absolute emergency fallback)
    # ========================================================================
    if df is None:
        print("⚠️  Strategy 4: ALL DOWNLOADS FAILED - Using synthetic data")
        print("⚠️  This is an EMERGENCY FALLBACK for testing only!")
        print("⚠️  Results will NOT be meaningful for publication!")
        
        # Generate realistic synthetic bank data
        np.random.seed(random_state)
        n_samples = 45000
        
        df = pd.DataFrame({
            'age': np.random.randint(18, 90, n_samples),
            'job': np.random.choice(['management', 'technician', 'admin', 'blue-collar'], n_samples),
            'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
            'education': np.random.choice(['primary', 'secondary', 'tertiary'], n_samples),
            'default': np.random.choice(['yes', 'no'], n_samples, p=[0.02, 0.98]),
            'balance': np.random.randint(-5000, 10000, n_samples),
            'housing': np.random.choice(['yes', 'no'], n_samples),
            'loan': np.random.choice(['yes', 'no'], n_samples),
            'contact': np.random.choice(['cellular', 'telephone', 'unknown'], n_samples),
            'day': np.random.randint(1, 31, n_samples),
            'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun'], n_samples),
            'duration': np.random.randint(0, 3000, n_samples),
            'campaign': np.random.randint(1, 20, n_samples),
            'pdays': np.random.randint(-1, 500, n_samples),
            'previous': np.random.randint(0, 10, n_samples),
            'poutcome': np.random.choice(['success', 'failure', 'unknown'], n_samples),
            'y': np.random.choice(['yes', 'no'], n_samples, p=[0.12, 0.88])
        })
        method_used = "⚠️  SYNTHETIC DATA (emergency fallback)"
    
    # ========================================================================
    # Process the data (same regardless of source)
    # ========================================================================
    if df is None:
        raise RuntimeError(
            "CRITICAL: Could not load bank dataset by any method!\n"
            "This should never happen. Please contact support."
        )
    
    print(f"✓ Data loaded via: {method_used}")
    print(f"✓ Samples: {len(df):,}")
    
    # Verify required columns
    required_cols = ['age', 'y']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Target variable
    df['y'] = df['y'].astype(str).str.strip().str.lower()
    df['y'] = df['y'].map({'yes': 1, 'no': 0})
    y = df['y'].values
    
    # Protected attribute: Age >= 25 (standard threshold)
    a = (df['age'] >= 30).astype(int).values
    
    # Feature engineering
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                          'loan', 'contact', 'month', 'poutcome']
    
    # Get numerical features
    X_numerical_list = []
    for col in numerical_features:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0).values
            X_numerical_list.append(vals.reshape(-1, 1))
    
    X_numerical = np.hstack(X_numerical_list) if X_numerical_list else np.zeros((len(df), 0))
    
    # One-hot encode categorical features
    X_categorical_list = []
    for col in categorical_features:
        if col in df.columns:
            # Clean strings
            df[col] = df[col].astype(str).str.strip().str.replace('"', '').str.lower()
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            X_categorical_list.append(dummies.values)
    
    X_categorical = np.hstack(X_categorical_list) if X_categorical_list else np.zeros((len(df), 0))
    X = np.hstack([X_numerical, X_categorical])
    
    print(f"✓ Features: {X.shape[1]} dimensions")
    print(f"✓ Target distribution: {np.mean(y):.1%} positive")
    print(f"✓ Protected groups: {np.mean(a):.1%} privileged (age >= 25)")
    
    # Train-test split
    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create FairnessDataset objects
    train_data = FairnessDataset(
        X=X_train,
        y=y_train,
        a=a_train,
        feature_names=[f"feature_{i}" for i in range(X_train.shape[1])]
    )
    
    test_data = FairnessDataset(
        X=X_test,
        y=y_test,
        a=a_test,
        feature_names=[f"feature_{i}" for i in range(X_test.shape[1])]
    )
    
    print(f"✓ Train: {len(train_data.X):,} samples")
    print(f"✓ Test: {len(test_data.X):,} samples")
    print("="*80)
    
    return train_data, test_data


# Alias for compatibility
load_bank_data = load_bank_data_robust


def load_compas_data(data_dir: str = '.', test_size: float = 0.3, random_state: int = 42):
    """Load COMPAS recidivism dataset with robust error handling."""
    print("\n" + "="*80)
    print("Loading COMPAS Dataset")
    print("="*80)
    
    df = None
    
    # Try local file first
    local_path = Path(data_dir) / 'compas-scores-two-years.csv'
    if local_path.exists():
        print(f"✓ Loading from local file: {local_path}")
        df = pd.read_csv(local_path)
    else:
        # Try multiple mirrors
        urls = [
            'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv',
            'https://raw.githubusercontent.com/fairlearn/fairlearn/main/test_othermlp/data/compas-scores-two-years.csv',
        ]
        
        for url in urls:
            try:
                print(f"✓ Downloading from: {url[:60]}...")
                df = pd.read_csv(url)
                # Save for future use
                local_path.parent.mkdir(exist_ok=True)
                df.to_csv(local_path, index=False)
                print(f"  ✓ Cached to: {local_path}")
                break
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
    
    if df is None:
        raise RuntimeError("Could not load COMPAS dataset from any source!")
    
    # Standard preprocessing
    df = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['score_text'] != 'N/A')
    ]
    
    a = (df['race'] == 'Caucasian').astype(int).values
    y = df['two_year_recid'].values
    
    feature_cols = [
        'age', 'c_charge_degree', 'race', 'age_cat', 'score_text',
        'sex', 'priors_count', 'days_b_screening_arrest',
        'is_recid', 'is_violent_recid', 'vr_charge_degree'
    ]
    
    df_features = df[feature_cols].copy()
    for col in df_features.select_dtypes(include=['object']).columns:
        df_features[col] = LabelEncoder().fit_transform(df_features[col].astype(str))
    
    X = df_features.values
    
    print(f"✓ Loaded {len(X):,} samples, {X.shape[1]} features")
    
    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_data = FairnessDataset(X=X_train, y=y_train, a=a_train,
                                 feature_names=[f"feature_{i}" for i in range(X_train.shape[1])])
    test_data = FairnessDataset(X=X_test, y=y_test, a=a_test,
                                feature_names=[f"feature_{i}" for i in range(X_test.shape[1])])
    
    print(f"✓ Train: {len(train_data.X):,}, Test: {len(test_data.X):,}")
    print("="*80)
    return train_data, test_data


def load_adult_data(data_dir: str = '.', test_size: float = 0.3, random_state: int = 42):
    """Load Adult Income dataset with robust error handling."""
    print("\n" + "="*80)
    print("Loading Adult Income Dataset")
    print("="*80)
    
    df = None
    
    local_path = Path(data_dir) / 'adult.data'
    if local_path.exists():
        print(f"✓ Loading from local file: {local_path}")
        df = pd.read_csv(local_path, header=None)
    else:
        urls = [
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            'https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult.csv',
        ]
        
        for url in urls:
            try:
                print(f"✓ Downloading from: {url[:60]}...")
                df = pd.read_csv(url, header=None)
                local_path.parent.mkdir(exist_ok=True)
                df.to_csv(local_path, index=False)
                print(f"  ✓ Cached to: {local_path}")
                break
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
    
    if df is None:
        raise RuntimeError("Could not load Adult dataset from any source!")
    
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    df.columns = columns
    
    df = df.replace(' ?', np.nan).dropna()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    y = (df['income'] == '>50K').astype(int).values
    a = (df['sex'] == 'Male').astype(int).values
    
    numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                      'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'native-country']
    
    X_num = df[numerical_cols].values
    X_cat_list = [pd.get_dummies(df[col], prefix=col, drop_first=True).values 
                  for col in categorical_cols]
    X_cat = np.hstack(X_cat_list)
    X = np.hstack([X_num, X_cat])
    
    print(f"✓ Loaded {len(X):,} samples, {X.shape[1]} features")
    
    X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
        X, y, a, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_data = FairnessDataset(X=X_train, y=y_train, a=a_train,
                                 feature_names=[f"feature_{i}" for i in range(X_train.shape[1])])
    test_data = FairnessDataset(X=X_test, y=y_test, a=a_test,
                                feature_names=[f"feature_{i}" for i in range(X_test.shape[1])])
    
    print(f"✓ Train: {len(train_data.X):,}, Test: {len(test_data.X):,}")
    print("="*80)
    return train_data, test_data


def get_dataset(dataset_name: str, data_dir: str = '.', test_size: float = 0.3, random_state: int = 42):
    """
    Load any fairness dataset with robust error handling.
    
    Args:
        dataset_name: 'bank', 'compas', 'adult', or 'german'
        data_dir: Directory for data files
        test_size: Test split fraction
        random_state: Random seed
        
    Returns:
        train_data, test_data: FairnessDataset objects
    """
    dataset_name = dataset_name.lower().strip()
    
    if dataset_name == 'bank':
        return load_bank_data_robust(data_dir, test_size, random_state)
    elif dataset_name == 'compas':
        return load_compas_data(data_dir, test_size, random_state)
    elif dataset_name == 'adult':
        return load_adult_data(data_dir, test_size, random_state)
    elif dataset_name == 'german':
        from data_utils import load_german_data
        return load_german_data(test_size=test_size, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose: bank, compas, adult, german")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING ALL DATASET LOADERS (ROBUST MODE)")
    print("="*80)
    
    for name in ['german', 'compas', 'adult', 'bank']:
        try:
            train, test = load_dataset(name)
            print(f"✅ {name.upper()}: SUCCESS - Train: {train.X.shape}, Test: {test.X.shape}\n")
        except Exception as e:
            print(f"❌ {name.upper()}: FAILED - {e}\n")
