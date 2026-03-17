# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import time
import os
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
import csv
import math
import argparse
import json

# Import imputation methods from Imputation_method.py
from Imputation_method import knn_imputation, mice_impute, rf_iterative_impute

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(config_path, dataset_name=None):
    """Load hyperparameters from config file, merging dataset-specific values with defaults."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Start with defaults
    hypers = config.get('default', {}).copy()
    
    # Merge with dataset-specific hypers if available
    if dataset_name and dataset_name in config:
        hypers.update(config[dataset_name])
    
    # Preserve global settings if they exist
    if 'global_settings' in config:
        hypers['global_settings'] = config['global_settings']
        
    return hypers

# ============================================================================

# --- Helper Functions ---

def generate_random_seeds(num_seeds=3, seed_range=(1, 10000)):
    """Generate unique random seeds for experiments"""
    import random
    random.seed(int(time.time()))
    seeds = []
    while len(seeds) < num_seeds:
        new_seed = random.randint(seed_range[0], seed_range[1])
        if new_seed not in seeds:
            seeds.append(new_seed)
    return seeds

class RobustSVMMinMax:
    def __init__(self, input_dim, n_samples, C_svm=1.0, lr_primal=0.01, lr_dual=1.0, fit_intercept=False):
        """
        Robust SVM with Stochastic Primal-Dual Repair (SPDR)
        """
        self.C_svm = C_svm
        self.lr_primal = lr_primal
        self.lr_dual = lr_dual
        self.fit_intercept = fit_intercept
        self.input_dim = input_dim
        self.n_samples = n_samples
        
        self.w = np.zeros(input_dim)
        self.b = 0.0
        self.alpha = np.zeros(n_samples)
        # Standard SVM regularization: lambda = 1 / (n * C)
        self.lmbda = 1.0 / (n_samples * C_svm) if C_svm > 0 else 1e-4
        self.is_fitted = False

    def train_step(self, X_adv, Y_i, indices):
        """Mini-batch Stochastic Primal-Dual update step."""
        margins = Y_i * (np.dot(X_adv, self.w) + self.b)
        margins = np.nan_to_num(margins, nan=0.0, posinf=1e6, neginf=-1e6)
        grad_dual = 1.0 - margins
        
        # Dual Gradient Clipping to prevent explosion on large datasets
        self.alpha[indices] = np.clip(self.alpha[indices] + self.lr_dual * np.clip(grad_dual, -1.0, 1.0), 0, 1.0)
        
        # Primal Gradient Scaling: Divide by batch size (len(indices))
        update_w = np.dot(self.alpha[indices] * Y_i, X_adv) / len(indices)
        update_w = np.nan_to_num(update_w, nan=0.0, posinf=1e6, neginf=-1e6)
        self.w = (1.0 - self.lr_primal * self.lmbda) * self.w + (self.lr_primal * update_w)
        self.w = np.nan_to_num(self.w, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if self.fit_intercept:
            self.b += self.lr_primal * np.mean(self.alpha[indices] * Y_i)
            self.b = float(np.nan_to_num(self.b, nan=0.0, posinf=1e6, neginf=-1e6))
        self.is_fitted = True

    def compute_sensitivity(self, bounds, mask):
        """Compute Lagrange Multiplier Sensitivity (Si) for AMR sample selection."""
        width = bounds[:, :, 1] - bounds[:, :, 0]
        uncertainty_contribution = np.sum(np.abs(self.w) * width * mask, axis=1)
        # S_i = (1/n) * alpha_i * sum_j |w_j| * width_j
        S = (1.0 / self.n_samples) * self.alpha * uncertainty_contribution
        return np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, X):
        if not self.is_fitted:
            return np.ones(len(X))
        decision_scores = np.dot(X, self.w) + self.b
        return np.where(decision_scores >= 0, 1, -1)

    def score(self, X, y):
        if not self.is_fitted:
            return 0.0
        y_binary = np.where(y > 0, 1, -1)
        y_pred = self.predict(X)
        return accuracy_score(y_binary, y_pred)

    def compute_robust_duality_gap(self, X_train, y_train, bounds, mask, penalty=10.0):
        """Compute the Robust Duality Gap with proper scaling."""
        n = self.n_samples
        X_adv = get_adversarial_repair(self.w, self.b, X_train, bounds, mask, y_train)
        
        # Primal Loss
        lmbda = self.lmbda
        margins = y_train * (np.dot(X_adv, self.w) + self.b)
        hinge_losses = np.maximum(0, 1 - margins)
        primal_loss = 0.5 * lmbda * np.dot(self.w, self.w) + np.mean(hinge_losses)
        
        # Dual Objective
        # Ensure we don't divide by zero if lmbda is very small
        lmbda_safe = max(lmbda, 1e-9)
        alpha_mean = np.mean(self.alpha)
        dual_grad_sum = np.dot(self.alpha * y_train, X_adv)
        dual_obj = alpha_mean - (1.0 / (2.0 * lmbda_safe * n**2)) * np.dot(dual_grad_sum, dual_grad_sum)
        
        # Constraint check: sum(alpha_i y_i) should be 0
        constraint_violation = np.mean(self.alpha * y_train)
        dual_obj_adjusted = dual_obj - penalty * np.abs(constraint_violation)
        
        return primal_loss, dual_obj_adjusted, primal_loss - dual_obj_adjusted

def get_adversarial_repair(w, b, data, bounds, mask, y):
    """Vectorized Adversarial Edge Inference."""
    y_expanded = y[:, np.newaxis]
    w_expanded = w[np.newaxis, :]
    direction = y_expanded * w_expanded
    x_edge = np.where(direction > 0, bounds[:, :, 0], bounds[:, :, 1])
    X_adv = np.where(mask, x_edge, data)
    return X_adv

def load_dataset_from_csv(csv_file_path, config=None, test_csv_path=None):
    """Load dataset and compute initial mask/bounds."""
    test_size = config.get('TEST_SIZE', 0.2) if config else 0.2
    random_state = config.get('RANDOM_STATE', 42) if config else 42
    
    dataset_name = os.path.basename(csv_file_path).replace('.csv', '')
    # SUSY files have a header row; fraud MCAR/MAR/MNAR variant files are generated with headers.
    is_susy = 'susy' in dataset_name.lower()
    is_fraud_variant = dataset_name.startswith('fraud_') and any(m in dataset_name.lower() for m in ['mcar', 'mar', 'mnar'])
    df = pd.read_csv(csv_file_path, header=0 if (is_susy or is_fraud_variant) else 'infer')
    
    # Handle dataset-specific preprocessing
    if 'breast' in dataset_name.lower():
        # Breast cancer: skip ID column 0, features 1-9, label 10
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values.astype(float)
        # Label mapping {2, 4} -> {-1, 1}
        unique_labels = np.unique(y[~np.isnan(y)])
        if set(unique_labels) == {2.0, 4.0}:
            y = np.where(y == 2.0, -1.0, 1.0)
    elif is_susy:
        # SUSY: column 0 is label, columns 1-18 are features
        X = df.iloc[:, 1:].values
        y = df.iloc[:, 0].values.astype(float)
    else:
        # Default: all columns except last are features, last is label
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.astype(float)
    
    # Standardize labels to {-1, 1} if they are {0, 1}
    unique_labels = np.unique(y[~np.isnan(y)])
    if set(unique_labels) == {0.0, 1.0}:
        y = np.where(y == 0.0, -1.0, 1.0)
        
    X = pd.DataFrame(X).replace('', np.nan).values.astype(float)
    
    # --- Ground Truth Loading Logic ---
    X_train_gt = None
    # Identify the base dataset name to find the OG file
    base_ds_map = {
        'malware': './Data_LL/malware_OG_train.csv',
        'tuadromd': './Data_LL/tuadromd_OG_train.csv',
        'default': './Data_LL/default_OG_train.csv',
        'credit_default': './Data_LL/default_OG_train.csv',
        'fraud': './Data_LL/fraud_OG_train.csv',
        'susy': './Data_LL/susy_OG_train.csv'
    }
    
    og_file_path = None
    for key, path in base_ds_map.items():
        if dataset_name.startswith(key):
            og_file_path = path
            break
            
    if og_file_path and os.path.exists(og_file_path):
        try:
            df_og = pd.read_csv(og_file_path, header=0 if 'susy' in dataset_name.lower() else 'infer')
            # Apply same feature extraction as X
            if 'breast' in dataset_name.lower():
                X_gt_raw = df_og.iloc[:, 1:-1].values
            elif 'susy' in dataset_name.lower():
                X_gt_raw = df_og.iloc[:, 1:].values
            else:
                X_gt_raw = df_og.iloc[:, :-1].values
            X_train_gt = pd.DataFrame(X_gt_raw).replace('', np.nan).values.astype(float)
            print("Loaded Ground Truth (OG) training set from: {}".format(og_file_path))
        except Exception as e:
            print("Warning: Could not load ground truth file {}: {}".format(og_file_path, e))
    # ----------------------------------

    if test_csv_path and os.path.exists(test_csv_path):
        # Synthetic dataset with external test set
        X_train = X
        y_train = y
        
        df_test = pd.read_csv(test_csv_path, header=0 if is_susy else 'infer')
        if 'breast' in dataset_name.lower():
             X_test = df_test.iloc[:, 1:-1].values
             y_test = df_test.iloc[:, -1].values.astype(float)
        elif is_susy:
             X_test = df_test.iloc[:, 1:].values
             y_test = df_test.iloc[:, 0].values.astype(float)
        else:
             X_test = df_test.iloc[:, :-1].values
             y_test = df_test.iloc[:, -1].values.astype(float)
        
        # Standardize test labels
        unique_test_labels = np.unique(y_test[~np.isnan(y_test)])
        if set(unique_test_labels) == {0.0, 1.0}:
            y_test = np.where(y_test == 0.0, -1.0, 1.0)
        elif set(unique_test_labels) == {2.0, 4.0}:
            y_test = np.where(y_test == 2.0, -1.0, 1.0)
            
        X_test = pd.DataFrame(X_test).replace('', np.nan).values.astype(float)
    else:
        # Real-world dataset or fallback to split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
            )
        # Note: If we split, X_train_gt also needs to be split with the same seed.
        # GT file may have one extra row relative to the MCAR file; align before splitting.
        if X_train_gt is not None:
            n_align = min(X_train_gt.shape[0], X.shape[0])
            X_gt_aligned = X_train_gt[:n_align]
            y_align = y[:n_align]
            X_train_gt, _, _, _ = train_test_split(
                X_gt_aligned, y_align, test_size=test_size, random_state=random_state,
                stratify=y_align if len(np.unique(y_align)) > 1 else None
                )
    
    mask_train = np.isnan(X_train)
    col_min = np.nanmin(X_train, axis=0)
    col_max = np.nanmax(X_train, axis=0)
    col_min = np.nan_to_num(col_min, nan=0.0)
    col_max = np.nan_to_num(col_max, nan=1.0)
    
    n_train, n_features = X_train.shape
    bounds_train = np.zeros((n_train, n_features, 2))
    for j in range(n_features):
        bounds_train[:, j, 0] = col_min[j]
        bounds_train[:, j, 1] = col_max[j]
    
    scaler = MinMaxScaler()
    complete_rows = ~np.isnan(X_train).any(axis=1)
    if np.sum(complete_rows) > 0:
        scaler.fit(X_train[complete_rows])
    else:
        temp_imputer = SimpleImputer(strategy='mean')
        scaler.fit(temp_imputer.fit_transform(X_train))
    
    col_min_scaled = scaler.transform(col_min.reshape(1, -1)).flatten()
    col_max_scaled = scaler.transform(col_max.reshape(1, -1)).flatten()
    for j in range(n_features):
        bounds_train[:, j, 0] = col_min_scaled[j]
        bounds_train[:, j, 1] = col_max_scaled[j]
    
    X_train_scaled = X_train.copy()
    non_nan_mask = ~np.isnan(X_train)
    if np.any(non_nan_mask):
        X_train_filled = np.nan_to_num(X_train)
        X_train_transformed = scaler.transform(X_train_filled)
        X_train_scaled[non_nan_mask] = X_train_transformed[non_nan_mask]
    
    # Scale X_train_gt if it exists
    if X_train_gt is not None:
        X_train_gt = scaler.transform(np.nan_to_num(X_train_gt))
    
    X_test_scaled = scaler.transform(np.nan_to_num(X_test))
    num_incomplete_train = int(np.sum(np.any(mask_train, axis=1)))
    return X_train_scaled, y_train, X_test_scaled, y_test, dataset_name, mask_train, bounds_train, num_incomplete_train, X_train_gt

def impute_examples_classification(X_current_state,
                        example_indices_to_impute,
                        imputation_method,
                        original_dataset_context=None, 
                        col_stats=None,
                        fitted_imputer=None,
                        random_seed=None,
                        X_train_gt=None): 
    """
    Imputes specified examples in X_current_state using the chosen method.
    Supports KNN, MICE, RF iterative imputation, and Ground Truth (GT).
    """
    X_new_state = X_current_state.copy()
    if example_indices_to_impute is None or len(example_indices_to_impute) == 0: 
        return X_new_state

    indices_to_impute_arr = np.array(list(example_indices_to_impute))
    if indices_to_impute_arr.size == 0:
        return X_new_state
        
    valid_indices = indices_to_impute_arr[indices_to_impute_arr < X_new_state.shape[0]]
    if imputation_method == 'gt' and X_train_gt is not None:
        valid_indices = valid_indices[valid_indices < X_train_gt.shape[0]]
    if valid_indices.size == 0:
        return X_new_state

    subset_to_impute = X_new_state[valid_indices]
    if subset_to_impute.size == 0: 
        return X_new_state

    imputed_subset = subset_to_impute.copy() 

    if imputation_method == 'gt':
        if X_train_gt is not None:
            # Match the ground truth values using the same indices
            # We must ensure that X_train_gt has the same row order as X_current_state
            imputed_subset = X_train_gt[valid_indices]
        else:
            print("Warning: Ground Truth requested but not available. Falling back to mean.")
            imputation_method = 'mean'

    if imputation_method == 'mean':
        if col_stats is None or 'means' not in col_stats:
            if original_dataset_context is None or original_dataset_context.shape[0] == 0:
                # Fallback: nan=0.5 for MinMaxScaler
                imputed_subset = np.nan_to_num(imputed_subset, nan=0.5)
            else:
                imputer = SimpleImputer(strategy='mean')
                # Only use complete rows for fitting imputer to avoid bias
                complete_rows = ~np.isnan(original_dataset_context).any(axis=1)
                if np.any(complete_rows):
                    imputer.fit(original_dataset_context[complete_rows])
                else:
                    imputer.fit(np.nan_to_num(original_dataset_context))
                imputed_subset = imputer.transform(imputed_subset) 
        else:
            col_means = col_stats['means']
            for i in range(imputed_subset.shape[0]):
                for feat_idx in range(imputed_subset.shape[1]):
                    if np.isnan(imputed_subset[i, feat_idx]):
                        if feat_idx < len(col_means):
                            imputed_subset[i, feat_idx] = col_means[feat_idx]
                        else:
                            imputed_subset[i, feat_idx] = 0.5
    
    elif imputation_method == 'knn':
        if fitted_imputer is not None:
            # Use pre-fitted KNN imputer
            imputed_subset, _ = knn_imputation(imputed_subset, "Transform", fitted_imputer)
        else:
            # Fallback to mean imputation if no fitted imputer available
            imputed_subset = np.nan_to_num(imputed_subset, nan=0.5)
    
    elif imputation_method == 'mice':
        if fitted_imputer is not None:
            # Use pre-fitted MICE imputer
            imputed_subset, _ = mice_impute(imputed_subset, condition="Transform", imputer=fitted_imputer)
        else:
            # Fallback to mean imputation if no fitted imputer available
            imputed_subset = np.nan_to_num(imputed_subset, nan=0.5)
    
    elif imputation_method == 'rf_iterative' or imputation_method == 'rf':
        if fitted_imputer is not None:
            # Use pre-fitted RF iterative imputer
            imputed_subset, _ = rf_iterative_impute(imputed_subset, condition="Transform", imputer=fitted_imputer)
        else:
            # Fallback to mean imputation if no fitted imputer available
            imputed_subset = np.nan_to_num(imputed_subset, nan=0.5)
    elif imputation_method != 'gt':
        # Default fallback
        imputed_subset = np.nan_to_num(imputed_subset, nan=0.5)

    X_new_state[valid_indices] = imputed_subset
    return X_new_state

def run_baseline_impute_all(X_train_inc, y_train, X_test, y_test, method='knn', config=None, X_train_gt=None):
    """Baseline: Impute all missing values and train standard SVM (SGDClassifier)."""
    start_time = time.time()
    
    c_svm = config.get('C_SVM', 1.0) if config else 1.0
    knn_neighbors = config.get('KNN_NEIGHBORS', 5) if config else 5
    random_state = config.get('RANDOM_STATE', 42) if config else 42
    
    # Standardize labels to {-1, 1}
    y_tr = np.where(np.array(y_train, dtype=float).ravel() > 0, 1, -1)
    y_te = np.where(np.array(y_test, dtype=float).ravel() > 0, 1, -1)

    # 1. Impute everything
    if method == "gt" and X_train_gt is not None:
        X_tr = np.array(X_train_gt, dtype=float)
        if X_tr.shape[0] != y_tr.shape[0]:
            min_len = min(X_tr.shape[0], y_tr.shape[0])
            X_tr = X_tr[:min_len]
            y_tr = y_tr[:min_len]
        X_te = X_test
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        X_tr = imputer.fit_transform(X_train_inc)
        X_te = imputer.transform(X_test)
    elif method == 'mice':
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(max_iter=config.get('MICE_MAX_ITER', 2), random_state=random_state)
        X_tr = imputer.fit_transform(X_train_inc)
        X_te = imputer.transform(X_test)
    elif method in ['rf', 'rf_iterative']:
        from Imputation_method import rf_iterative_impute
        # rf_iterative_impute expects Fit/Transform condition
        X_tr, imputer = rf_iterative_impute(X_train_inc, condition="Fit")
        X_te, _ = rf_iterative_impute(X_test, condition="Transform", imputer=imputer)
    else:
        imputer = SimpleImputer(strategy='mean')
        X_tr = imputer.fit_transform(X_train_inc)
        X_te = imputer.transform(X_test)
        
    # 2. Train standard SVM using SGDClassifier (hinge loss)
    # Standard SVM regularization: alpha = 1 / (n * C)
    alpha_reg = 1.0 / (len(X_tr) * c_svm) if c_svm > 0 else 0.0001
    model = SGDClassifier(loss='hinge', alpha=alpha_reg, fit_intercept=config.get('FIT_INTERCEPT', True), 
                          max_iter=1000, tol=1e-3, random_state=random_state)
    model.fit(X_tr, y_tr)
    
    acc = model.score(X_te, y_te)
    return acc, time.time() - start_time

def run_single_experiment(original_dataset, labels, seed=None, 
                          X_test_eval=None, y_test_eval=None, 
                          X_initial_dirty_for_eval_context=None, 
                          X_train_gt=None, 
                          convergence_log_path=None,
                          imputation_method='mean',
                          mask_train=None,
                          bounds_train=None,
                          config=None,
                          num_incomplete_train=None): 
    """Single run of SPDR ACM algorithm"""
    total_amr_time = 0.0
    try:
        # --- Baseline Method Implementation (Impute All with GT) ---
        if imputation_method == 'baseline':
            if X_train_gt is None:
                print("Error: Baseline requested but X_train_gt is None.")
                return {'total_time': 0.0, 'total_imputed_count': 0, 'imputation_ratio': 0.0, 'test_accuracy': 0.0, 'uncertainty_range': 0.0, 'total_epochs': 0, 'imputed_indices': []}
            
            s_baseline = time.time()
            # Use X_train_gt as the training data (this is the originally complete dataset)
            X_train_full = X_train_gt
            y_train_full = np.where(np.array(labels, dtype=float).ravel() > 0, 1, -1)
            if X_train_full.shape[0] != y_train_full.shape[0]:
                min_len = min(X_train_full.shape[0], y_train_full.shape[0])
                print("Warning: Baseline length mismatch (X={}, y={}). Truncating both to {} for alignment.".format(
                    X_train_full.shape[0], y_train_full.shape[0], min_len
                ))
                X_train_full = X_train_full[:min_len]
                y_train_full = y_train_full[:min_len]
            
            # Use sklearn SGDClassifier for quick evaluation (equivalent to SVM with hinge loss)
            # We use the same hyperparameters where possible
            C_svm = config.get('C_SVM', 1.0)
            alpha = 1.0 / (len(X_train_full) * C_svm) if C_svm > 0 else 0.0001
            
            clf = SGDClassifier(loss='hinge', alpha=alpha, fit_intercept=config.get('FIT_INTERCEPT', True), 
                                max_iter=config.get('MAX_TOTAL_EPOCHS', 50), random_state=seed)
            clf.fit(X_train_full, y_train_full)
            
            test_acc = clf.score(X_test_eval, np.where(y_test_eval > 0, 1, -1))
            total_time = time.time() - s_baseline
            
            print(f"Baseline (Full GT) completed in {total_time:.4f}s. Test Accuracy: {test_acc:.4f}")
            
            return {
                'total_time': total_time,
                'total_imputed_count': num_incomplete_train,
                'imputation_ratio': 1.0,
                'test_accuracy': test_acc,
                'uncertainty_range': 0.0,
                'total_epochs': config.get('MAX_TOTAL_EPOCHS', 50),
                'imputed_indices': []
            }
        # -----------------------------------------------------------

        # Use config if provided, otherwise fallback to defaults
        C_svm = config.get('C_SVM', 1.0)
        max_total_epochs = config.get('MAX_TOTAL_EPOCHS', 50)
        lr_primal = config.get('LR_PRIMAL', config.get('STEP_SIZE', 0.1))
        lr_dual = config.get('LR_DUAL', 1.0)
        fit_intercept = config.get('FIT_INTERCEPT', True)
        batch_size = config.get('BATCH_SIZE', 64)
        train_epochs_per_iter = config.get('TRAIN_EPOCHS_PER_ITER', 1)
        select_k_samples = config.get('SELECT_K_SAMPLES', 10)
        loss_convergence_threshold = config.get('LOSS_CONVERGENCE_THRESHOLD', 0.05)
        repair_threshold_divisor = config.get('REPAIR_THRESHOLD_DIVISOR', 100.0)
        dual_penalty = config.get('DUAL_CONSTRAINT_PENALTY', 10.0)
        
        if seed is not None: np.random.seed(seed)
        
        X_current = np.array(original_dataset, dtype=float)
        y_labels = np.where(np.array(labels, dtype=float).ravel() > 0, 1, -1)
        n_samples, n_features = X_current.shape
        
        robust_svm = RobustSVMMinMax(input_dim=n_features, n_samples=n_samples, C_svm=C_svm, 
                                     lr_primal=lr_primal, lr_dual=lr_dual, fit_intercept=fit_intercept)
        # Neutral dual initialization improves stability and avoids all-zero sensitivity at startup.
        robust_svm.alpha = np.full(n_samples, 0.5)
        imputed_indices = set()
        total_imputed_samples = 0
        current_mask = mask_train.copy() if mask_train is not None else np.isnan(X_current)
        current_bounds = bounds_train.copy()
        
        if num_incomplete_train is None:
            num_incomplete_train = int(np.sum(np.any(current_mask, axis=1)))
        
        # START CORE TIMER (Moved up to include imputer fitting)
        s_core_init = time.time()

        # --- High-Quality Warm-Start ---
        # Train on clean data only to get a good starting point
        clean_indices = np.where(~current_mask.any(axis=1))[0]
        if len(clean_indices) > 0:
            print(f"Warm-starting SVM with {len(clean_indices)} clean samples...")
            X_clean = X_current[clean_indices]
            y_clean = y_labels[clean_indices]
            if len(np.unique(y_clean)) > 1:
                try:
                    # Use hinge loss SGDClassifier for SVM warm-start
                    lmbda = 1.0 / (n_samples * C_svm) if C_svm > 0 else 1e-4
                    warm_start_clf = SGDClassifier(loss='hinge', alpha=lmbda, 
                                                  fit_intercept=fit_intercept, 
                                                  max_iter=10, random_state=seed)
                    warm_start_clf.fit(X_clean, y_clean)
                    
                    robust_svm.w = np.nan_to_num(warm_start_clf.coef_[0].copy(), nan=0.0, posinf=1e6, neginf=-1e6)
                    if fit_intercept:
                        robust_svm.b = float(np.nan_to_num(warm_start_clf.intercept_[0], nan=0.0, posinf=1e6, neginf=-1e6))
                    robust_svm.is_fitted = True
                except Exception as warm_start_error:
                    print("Warning: Warm-start failed, continuing with default initialization: {}".format(warm_start_error))
        # -------------------------------

        # Setup imputer for selection-based repair
        fitted_imputer = None
        knn_neighbors = config.get('KNN_NEIGHBORS', 5)
        mice_max_iter = config.get('MICE_MAX_ITER', 2)
        
        if imputation_method == 'knn':
            fitted_imputer, _ = knn_imputation(np.nan_to_num(X_current), "Fit", None, knn_neighbors)
        elif imputation_method == 'mice':
            fitted_imputer, _ = mice_impute(np.nan_to_num(X_current), mice_max_iter, seed, "Fit", None)
        elif imputation_method == 'rf_iterative' or imputation_method == 'rf':
            fitted_imputer, _ = rf_iterative_impute(np.nan_to_num(X_current), condition="Fit")

        total_amr_time += (time.time() - s_core_init)

        test_acc = 0.0
        
        # Setup per-epoch logging
        conv_file = None
        conv_writer = None
        if convergence_log_path:
            conv_file = open(convergence_log_path, 'w', newline='')
            conv_writer = csv.DictWriter(conv_file, fieldnames=['epoch', 'primal_loss', 'dual_obj', 'duality_gap', 'sum_S', 'test_accuracy', 'imputed_this_epoch'])
            conv_writer.writeheader()

        final_epoch = 0
        for epoch in range(max_total_epochs):
            # START CORE TIMER
            s_core = time.time()
            
            # 1. Sensitivity finding (Ambiguity Evaluation)
            # Calculated at start so repairs happen before the current epoch's training
            S = robust_svm.compute_sensitivity(current_bounds, current_mask)
            sum_S = float(np.nan_to_num(np.sum(S), nan=np.inf, posinf=np.inf, neginf=np.inf))
            
            # 2. Repair Phase: Select samples exceeding sensitivity threshold
            imputed_this_epoch = 0
            candidate_indices = np.where((~np.isin(np.arange(n_samples), list(imputed_indices))) & (current_mask.any(axis=1)))[0]
            
            if len(candidate_indices) > 0:
                candidate_S = S[candidate_indices]
                # Threshold-first selection: only repair if S_i > threshold
                selection_threshold = loss_convergence_threshold / repair_threshold_divisor
                valid_candidates_mask = candidate_S > selection_threshold
                
                if np.any(valid_candidates_mask):
                    valid_candidates = candidate_indices[valid_candidates_mask]
                    valid_S = candidate_S[valid_candidates_mask]
                    
                    # Cap by select_k_samples
                    num_to_repair = min(len(valid_candidates), select_k_samples)
                    top_k_idx = np.argsort(-valid_S)[:num_to_repair]
                    actual_indices_to_repair = valid_candidates[top_k_idx]
                    
                    # Reveal selected samples using chosen imputation method
                    X_current = impute_examples_classification(
                        X_current, actual_indices_to_repair, imputation_method,
                        original_dataset_context=original_dataset,
                        fitted_imputer=fitted_imputer,
                        random_seed=seed,
                        X_train_gt=X_train_gt
                    )
                    for r_idx in actual_indices_to_repair:
                        imputed_indices.add(r_idx)
                        total_imputed_samples += 1
                        imputed_this_epoch += 1
                        
                        # Update bounds and mask for repaired samples
                        repaired_val = X_current[r_idx]
                        current_bounds[r_idx, :, 0] = repaired_val
                        current_bounds[r_idx, :, 1] = repaired_val
                        current_mask[r_idx, :] = False
            
            # 3. Training Phase
            # Now training uses the newly repaired samples immediately
            for _ in range(train_epochs_per_iter):
                indices = np.random.permutation(n_samples)
                for start_idx in range(0, n_samples, batch_size):
                    batch_idx = indices[start_idx : start_idx + batch_size]
                    X_batch = X_current[batch_idx]
                    Y_batch = y_labels[batch_idx]
                    Bounds_batch = current_bounds[batch_idx]
                    Mask_batch = current_mask[batch_idx]
                    
                    X_adv = get_adversarial_repair(robust_svm.w, robust_svm.b, X_batch, Bounds_batch, Mask_batch, Y_batch)
                    robust_svm.train_step(X_adv, Y_batch, batch_idx)
            
            # STOP CORE TIMER
            total_amr_time += (time.time() - s_core)
            
            # --- DIAGNOSTIC SECTION (NOT TIMED) ---
            primal, dual, gap = robust_svm.compute_robust_duality_gap(X_current, y_labels, current_bounds, current_mask, penalty=dual_penalty)
            test_acc = robust_svm.score(X_test_eval, y_test_eval)
            
            if conv_writer:
                conv_writer.writerow({
                    'epoch': epoch + 1,
                    'primal_loss': primal,
                    'dual_obj': dual,
                    'duality_gap': gap,
                    'sum_S': sum_S,
                    'test_accuracy': test_acc,
                    'imputed_this_epoch': imputed_this_epoch
                })
                conv_file.flush()

            final_epoch = epoch + 1
            # Convergence: total sensitivity below threshold means no incomplete sample
            # contributes meaningfully to the decision boundary uncertainty.
            if sum_S < loss_convergence_threshold: 
                print("Converged at epoch {} with sum_S {:.6f}".format(final_epoch, sum_S))
                break
            
        if conv_file: conv_file.close()
        
        imputation_ratio = total_imputed_samples / num_incomplete_train if num_incomplete_train > 0 else 0.0
        
        return {
            'total_time': total_amr_time,
            'total_imputed_count': total_imputed_samples,
            'imputation_ratio': imputation_ratio,
            'test_accuracy': test_acc,
            'uncertainty_range': np.mean(S) if 'S' in locals() else 0.0,
            'total_epochs': final_epoch,
            'imputed_indices': list(imputed_indices)
        }
    except Exception as e:
        print("Error in single experiment: {}".format(e))
        import traceback; traceback.print_exc()
        return {'total_time': 0.0, 'total_imputed_count': 0, 'imputation_ratio': 0.0, 'test_accuracy': 0.0, 'uncertainty_range': 0.0, 'total_epochs': 0, 'imputed_indices': []}

def findminimalImputation(original_dataset, labels, seed=None, 
                          X_test_eval=None, y_test_eval=None, 
                          X_initial_dirty_for_eval_context=None, 
                          original_dataset_complete_for_gt_imputation=None, 
                          log_file_path=None,
                          imputation_method='mean',
                          mask_train=None,
                          bounds_train=None,
                          config=None,
                          num_incomplete_train=None): 
    """SPDR-based ACM algorithm with multiple runs and averaging"""
    print("Starting Stochastic Primal-Dual Repair (SPDR) ACM Algorithm")
    num_random_seeds = config.get('NUM_RANDOM_SEEDS', 3) if config else 3
    seeds = generate_random_seeds(num_random_seeds)
    results = []
    
    results_output_dir = './ACM_Results/Multi_Run_Logs'
    os.makedirs(results_output_dir, exist_ok=True)
    epoch_log_file_path = os.path.join(results_output_dir, 'epoch_log.csv')
    
    if mask_train is None: mask_train = np.isnan(original_dataset)
    if num_incomplete_train is None:
        num_incomplete_train = int(np.sum(np.any(mask_train, axis=1)))
    
    if bounds_train is None:
        c_min = np.nan_to_num(np.nanmin(original_dataset, axis=0), nan=0.0)
        c_max = np.nan_to_num(np.nanmax(original_dataset, axis=0), nan=1.0)
        bounds_train = np.zeros((original_dataset.shape[0], original_dataset.shape[1], 2))
        for j in range(original_dataset.shape[1]):
            bounds_train[:, j, 0] = c_min[j]; bounds_train[:, j, 1] = c_max[j]

    with open(epoch_log_file_path, 'w', newline='') as epoch_log_file:
        epoch_log_writer = csv.DictWriter(epoch_log_file, fieldnames=['run', 'total_time', 'total_imputed_count', 'imputation_ratio', 'test_accuracy', 'uncertainty_range', 'total_epochs'])
        epoch_log_writer.writeheader()
        
        for i, run_seed in enumerate(seeds):
            print("Run {}/{} with seed {}".format(i+1, len(seeds), run_seed))
            result = run_single_experiment(
                original_dataset, labels, seed=run_seed, X_test_eval=X_test_eval, y_test_eval=y_test_eval,
                X_train_gt=original_dataset_complete_for_gt_imputation,
                imputation_method=imputation_method, mask_train=mask_train.copy(), bounds_train=bounds_train.copy(),
                config=config, num_incomplete_train=num_incomplete_train
            )
            results.append(result)
            row_to_write = {'run': i+1}
            for k, v in result.items():
                if k != 'imputed_indices':
                    row_to_write[k] = v
            epoch_log_writer.writerow(row_to_write)
        
        metrics = ['total_time', 'total_imputed_count', 'imputation_ratio', 'test_accuracy', 'uncertainty_range', 'total_epochs']
        avg_result = {m: "{:.4f} ± {:.4f}".format(np.mean([r[m] for r in results]), np.std([r[m] for r in results], ddof=1)) for m in metrics}
        avg_row = {'run': 'average'}
        for k, v in avg_result.items():
            avg_row[k] = v
        epoch_log_writer.writerow(avg_row)
    
    final_result = results[-1]
    return final_result['imputed_indices'], final_result['imputed_indices'], final_result['test_accuracy']

def run_acm_on_csv_file_with_imputation_method(csv_file_path, imputation_method, output_dir=None, seeds_to_try=None, config=None, test_csv_path=None):
    if config is None:
        config = load_config('config_svm.json')
    
    if output_dir is None:
        output_dir = config.get('global_settings', {}).get('output_dir', './ACM_Results')

    if seeds_to_try is None:
        num_random_seeds = config.get('NUM_RANDOM_SEEDS', 3)
        seeds_to_try = generate_random_seeds(num_random_seeds)
        
    X_train_inc, y_train, X_test, y_test, dataset_name, mask_train, bounds_train, num_incomplete_train, X_train_gt = load_dataset_from_csv(csv_file_path, config=config, test_csv_path=test_csv_path)
    
    # Reload config with base dataset name to get hyperparameters
    base_dataset_name = dataset_name.split('_')[0]
    if base_dataset_name == 'default': base_dataset_name = 'credit_default'
    config = load_config('config_svm.json', dataset_name=base_dataset_name)
    
    # Check if this is a special dataset for exporting training data
    special_datasets = config.get('global_settings', {}).get('special_datasets', [])
    is_special = any(ds.lower() in dataset_name.lower() for ds in special_datasets)
    
    run_impute_all_baseline = config.get('global_settings', {}).get('run_impute_all_baseline', True)
    if run_impute_all_baseline:
        # Run Baseline reference metric
        if imputation_method == 'baseline' and X_train_gt is not None:
            s_baseline = time.time()
            y_train_full = np.where(np.array(y_train, dtype=float).ravel() > 0, 1, -1)
            X_train_full = np.array(X_train_gt, dtype=float)
            if X_train_full.shape[0] != y_train_full.shape[0]:
                min_len = min(X_train_full.shape[0], y_train_full.shape[0])
                X_train_full = X_train_full[:min_len]
                y_train_full = y_train_full[:min_len]
            
            c_svm = config.get('C_SVM', 1.0)
            alpha_reg = 1.0 / (len(X_train_full) * c_svm) if c_svm > 0 else 0.0001
            baseline_model = SGDClassifier(loss='hinge', alpha=alpha_reg, fit_intercept=config.get('FIT_INTERCEPT', True), 
                                          max_iter=1000, tol=1e-3, random_state=config.get('RANDOM_STATE', 42))
            baseline_model.fit(X_train_full, y_train_full)
            baseline_acc = baseline_model.score(X_test, np.where(y_test > 0, 1, -1))
            baseline_time = time.time() - s_baseline
            print(f"\nBaseline (Full GT): Accuracy = {baseline_acc:.4f}, Time = {baseline_time:.4f}s")
        else:
            baseline_acc, baseline_time = run_baseline_impute_all(X_train_inc, y_train, X_test, y_test, method=imputation_method, config=config, X_train_gt=X_train_gt)
            print(f"\nBaseline (Impute All {imputation_method}): Accuracy = {baseline_acc:.4f}, Time = {baseline_time:.4f}s")
    else:
        baseline_acc = np.nan
        baseline_time = 0.0
        print("\nBaseline (Impute-All reference): Skipped by config (run_impute_all_baseline=false)")

    iter_log_dir = os.path.join(output_dir, 'Iter_Logs')
    results_output_dir = os.path.join(output_dir, 'Final_Results')
    os.makedirs(iter_log_dir, exist_ok=True); os.makedirs(results_output_dir, exist_ok=True)
    
    results = []
    epoch_log_file_path = os.path.join(results_output_dir, '{}_{}_epoch_log.csv'.format(imputation_method, dataset_name))
    
    with open(epoch_log_file_path, 'w', newline='') as epoch_log_file:
        epoch_log_writer = csv.DictWriter(epoch_log_file, fieldnames=['run', 'total_time', 'total_imputed_count', 'imputation_ratio', 'test_accuracy', 'baseline_accuracy', 'baseline_time', 'uncertainty_range', 'total_epochs'])
        epoch_log_writer.writeheader()
        
        for i, run_seed in enumerate(seeds_to_try):
            print("\nMethod: {}, Run {}/{} with seed {}".format(imputation_method, i+1, len(seeds_to_try), run_seed))
            # Per-trial convergence log
            iter_conv_log_path = os.path.join(iter_log_dir, '{}_{}_run{}_convergence.csv'.format(imputation_method, dataset_name, i+1))
            
            result = run_single_experiment(X_train_inc, y_train, seed=run_seed, X_test_eval=X_test, y_test_eval=y_test, 
                                            X_train_gt=X_train_gt,
                                            convergence_log_path=iter_conv_log_path,
                                            imputation_method=imputation_method, mask_train=mask_train, bounds_train=bounds_train,
                                            config=config, num_incomplete_train=num_incomplete_train)
            results.append(result)
            row_to_write = {'run': i+1, 'baseline_accuracy': baseline_acc, 'baseline_time': baseline_time}
            for k, v in result.items():
                if k != 'imputed_indices':
                    row_to_write[k] = v
            epoch_log_writer.writerow(row_to_write)
        
            # Export partially imputed training set for special datasets
            if is_special:
                try:
                    export_path = os.path.join(results_output_dir, f"{dataset_name}_{imputation_method}_partially_imputed_train_run{i+1}.csv")
                    df_special = pd.read_csv(csv_file_path)
                    
                    # For synthetic datasets with external test set, we don't split
                    if test_csv_path:
                        train_df = df_special
                    else:
                        test_size = config.get('TEST_SIZE', 0.2)
                        random_state = config.get('RANDOM_STATE', 42)
                        train_df, _ = train_test_split(df_special, test_size=test_size, random_state=random_state, 
                                                       stratify=df_special.iloc[:, -1] if len(np.unique(df_special.iloc[:, -1][~np.isnan(df_special.iloc[:, -1])])) > 1 else None)
                    
                    # Create a copy to modify
                    train_export = train_df.copy().astype(object)
                    imputed_idx_list = result.get('imputed_indices', [])
                    
                    # Keep track of indices to keep: all complete rows + AMR-selected incomplete rows
                    # mask_train is True where values are missing
                    is_incomplete = mask_train.any(axis=1)
                    complete_indices = np.where(~is_incomplete)[0]
                    indices_to_keep = sorted(list(complete_indices) + list(imputed_idx_list))
                    
                    # Positional repair for selected incomplete samples
                    for local_idx in imputed_idx_list:
                        # For each feature that was NaN in this row, mark as empty string
                        row_mask = mask_train[local_idx]
                        if np.any(row_mask):
                            for col_idx in range(len(row_mask)):
                                if row_mask[col_idx]:
                                    train_export.iloc[local_idx, col_idx] = ""
                    
                    # Filter the export dataframe to only include indices_to_keep
                    train_export_filtered = train_export.iloc[indices_to_keep]
                    
                    train_export_filtered.to_csv(export_path, index=False)
                    print(f"Special dataset training set exported to: {export_path} (Rows: {len(train_export_filtered)})")
                except Exception as e:
                    print(f"Error exporting special dataset CSV: {e}")
        
        metrics = ['total_time', 'total_imputed_count', 'imputation_ratio', 'test_accuracy', 'uncertainty_range', 'total_epochs']
        avg_result = {m: "{:.4f} ± {:.4f}".format(np.mean([r[m] for r in results]), np.std([r[m] for r in results], ddof=1)) for m in metrics}
        avg_row = {'run': 'average', 'baseline_accuracy': baseline_acc, 'baseline_time': baseline_time}
        for k, v in avg_result.items():
            avg_row[k] = v
        epoch_log_writer.writerow(avg_row)
    
    return {'dataset_name': dataset_name, 'imputation_method': imputation_method, 'results': results, 'final_result': results[-1], 'missing_factor': np.mean(mask_train)}

def run_acm_on_csv_file(csv_file_path, output_dir=None, seeds_to_try=None, imputation_methods=None, config=None, test_csv_path=None):
    if config is None:
        # Initial load to get general defaults and imputation methods
        config = load_config('config_svm.json')
    
    if output_dir is None:
        output_dir = config.get('global_settings', {}).get('output_dir', './ACM_Results')
        
    if seeds_to_try is None:
        num_random_seeds = config.get('NUM_RANDOM_SEEDS', 3)
        seeds_to_try = generate_random_seeds(num_random_seeds)
    
    if imputation_methods is None:
        imputation_methods = config.get('global_settings', {}).get('imputation_methods', config.get('IMPUTATION_METHODS', ['knn']))
        
    dataset_name = os.path.basename(csv_file_path).replace('.csv', '')
    all_results = {}
    for imp_method in imputation_methods:
        try:
            result = run_acm_on_csv_file_with_imputation_method(csv_file_path, imp_method, output_dir, seeds_to_try, config=config, test_csv_path=test_csv_path)
            all_results[imp_method] = result
        except Exception as e: print("Error in {} with {}: {}".format(dataset_name, imp_method, e))
    return {'dataset_name': dataset_name, 'imputation_results': all_results}

def run_acm_on_dataset_directory(dataset_dir=None, output_dir='./ACM_Results', seeds_to_try=None, file_pattern='*.csv', imputation_methods=None, config=None):
    if config is None:
        config = load_config('config_svm.json')
    
    if dataset_dir is None:
        dataset_dir = config.get('DEFAULT_DATASET_DIR', './Data_LL')
    import glob
    csv_files = sorted(glob.glob(os.path.join(dataset_dir, file_pattern)))
    all_results = {}
    for csv_file in csv_files:
        result = run_acm_on_csv_file(csv_file, output_dir, seeds_to_try, imputation_methods, config=config)
        all_results[result['dataset_name']] = result
    return all_results

def sanity_check(X, minimal_imputation_examples):
    missing_rows = np.where(np.isnan(X).any(axis=1))[0]
    examples_saved = [example for example in missing_rows if example not in minimal_imputation_examples]
    return len(examples_saved)

def mean_imputation(X_train, X_test, y_train, y_test, seed):
    start_time = time.time()
    imputer = SimpleImputer(strategy='mean')
    X_tr = imputer.fit_transform(X_train)
    X_te = imputer.transform(X_test)
    model = SGDClassifier(loss='hinge', max_iter=10000, tol=1e-3, random_state=seed)
    model.fit(X_tr, y_train)
    return model.score(X_te, y_test), f1_score(y_test, model.predict(X_te), zero_division=0), time.time() - start_time

def run_acm_algorithm(X_train_inc, y_train, X_test, y_test, output_dir='./ACM_Results', experiment_name='ACM_Experiment', seeds_to_try=None, imputation_methods=None, config=None):
    """Run AMR across imputation methods and seeds; report mean accuracy."""
    if config is None:
        config = load_config('config_svm.json')
    if seeds_to_try is None:
        num_random_seeds = config.get('NUM_RANDOM_SEEDS', 3)
        seeds_to_try = generate_random_seeds(num_random_seeds)
    if imputation_methods is None:
        imputation_methods = config.get('IMPUTATION_METHODS', ['knn'])
    all_imputation_results = {}
    for imp_method in imputation_methods:
        seed_accs = []
        last_imp_idx = []
        for seed_val in seeds_to_try:
            _, imp_idx, acc = findminimalImputation(X_train_inc, y_train, seed=seed_val, X_test_eval=X_test, y_test_eval=y_test, imputation_method=imp_method, config=config)
            seed_accs.append(acc)
            last_imp_idx = imp_idx
        mean_acc = float(np.mean(seed_accs))
        std_acc  = float(np.std(seed_accs, ddof=1)) if len(seed_accs) > 1 else 0.0
        all_imputation_results[imp_method] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'imputed_indices': last_imp_idx,
        }
    return all_imputation_results

# Dataset paths mapping
DATASET_CONFIGS = {
    'malware': {'default_file': './Data_LL/malware_MNAR_train_60.csv', 'test_file': './Data_LL/malware_test.csv', 'variants_dir': './Data_LL', 'prefix': 'malware'},
    'tuadromd': {'default_file': './Data_LL/tuadromd_MNAR_train_60.csv', 'test_file': './Data_LL/tuadromd_test.csv', 'variants_dir': './Data_LL', 'prefix': 'tuadromd'},
    'credit_default': {'default_file': './Data_LL/default_MNAR_train_60.csv', 'test_file': './Data_LL/default_test.csv', 'variants_dir': './Data_LL', 'prefix': 'default'},
    'fraud': {'default_file': './Data_LL/fraud_MCAR_train_20.csv', 'test_file': './Data_LL/fraud_test.csv', 'variants_dir': './Data_LL', 'prefix': 'fraud'},
    'susy': {'default_file': './Data_LL/susy_MCAR_train_20.csv', 'test_file': './Data_LL/susy_test.csv', 'variants_dir': './Data_LL', 'prefix': 'susy'},
    'breast': {'default_file': './datasets/breast.csv'},
    'water': {'default_file': './datasets/water.csv'},
    'online': {'default_file': './datasets/online.csv'},
    'bankruptcy': {'default_file': './datasets/bankrupt_normalized.csv'}
}

def parse_args():
    parser = argparse.ArgumentParser(description='SPDR Primal-Dual ACM Algorithm')
    parser.add_argument('--dataset', type=str, nargs='+', help='Dataset name(s)')
    parser.add_argument('--single', type=str, help='Path to a single CSV file')
    parser.add_argument('--directory', type=str, help='Path to a directory of CSV files')
    parser.add_argument('--output', type=str, default='./ACM_Results', help='Output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    config = load_config('config_svm.json')
    global_settings = config.get('global_settings', {})
    
    # Priority: Command line > Config > Default
    output_dir = args.output if args.output != './ACM_Results' else global_settings.get('output_dir', './ACM_Results')
    
    if args.single:
        print(f"Running single experiment on: {args.single}")
        run_acm_on_csv_file(args.single, output_dir, config=config)
    elif args.dataset:
        for ds_name in args.dataset:
            # Check for synthetic variants
            found_variant = False
            for base_ds in ["malware", "tuadromd", "credit_default", "default", "fraud", "susy"]:
                if ds_name.startswith(base_ds + "_"):
                    found_variant = True
                    break
            
            if found_variant:
                # Handle variant
                is_variant = False
                for base_ds, info in DATASET_CONFIGS.items():
                    # Match if ds_name starts with base_ds_ OR if base_ds is credit_default and ds_name starts with default_
                    if "variants_dir" in info and (ds_name.startswith(base_ds + "_") or (base_ds == 'credit_default' and ds_name.startswith("default_"))):
                        parts = ds_name.split('_')
                        if len(parts) >= 3:
                            mechanism = parts[-2]
                            factor = parts[-1]
                            prefix = info["prefix"]
                            csv_file = os.path.join(info["variants_dir"], f"{prefix}_{mechanism}_train_{factor}.csv")
                            test_csv_path = info["test_file"]
                            print(f"\n{'='*60}\nRunning experiment on variant: {ds_name}\n{'='*60}")
                            print(f"File: {csv_file}")
                            ds_config = load_config('config_svm.json', dataset_name=base_ds)
                            run_acm_on_csv_file(csv_file, output_dir, config=ds_config, test_csv_path=test_csv_path)
                            is_variant = True
                            break
                if not is_variant:
                    print(f"Warning: Variant {ds_name} not found or invalid.")
            elif ds_name in DATASET_CONFIGS:
                csv_file = DATASET_CONFIGS[ds_name]['default_file']
                test_csv_path = DATASET_CONFIGS[ds_name].get('test_file')
                print(f"\n{'='*60}\nRunning experiment on dataset: {ds_name}\n{'='*60}")
                print(f"File: {csv_file}")
                ds_config = load_config('config_svm.json', dataset_name=ds_name)
                run_acm_on_csv_file(csv_file, output_dir, config=ds_config, test_csv_path=test_csv_path)
            else:
                print(f"Warning: Dataset {ds_name} not found in DATASET_CONFIGS.")
    elif args.directory:
        print(f"Running directory experiment on: {args.directory}")
        run_acm_on_dataset_directory(args.directory, output_dir, config=config)
    else:
        # Default fallback: run datasets specified in config global_settings
        datasets_to_run = global_settings.get('datasets_to_run', [])
        if datasets_to_run:
            for ds_name in datasets_to_run:
                # Check if it's a synthetic variant
                is_variant = False
                for base_ds, info in DATASET_CONFIGS.items():
                    if "variants_dir" in info and (ds_name.startswith(base_ds + "_") or (base_ds == 'credit_default' and ds_name.startswith("default_"))):
                        parts = ds_name.split('_')
                        if len(parts) >= 3:
                            mechanism = parts[-2]
                            factor = parts[-1]
                            prefix = info["prefix"]
                            csv_file = os.path.join(info["variants_dir"], f"{prefix}_{mechanism}_train_{factor}.csv")
                            test_csv_path = info["test_file"]
                            print(f"\n{'='*60}\nRunning experiment on variant: {ds_name}\n{'='*60}")
                            print(f"File: {csv_file}")
                            ds_config = load_config('config_svm.json', dataset_name=base_ds)
                            run_acm_on_csv_file(csv_file, output_dir, config=ds_config, test_csv_path=test_csv_path)
                            is_variant = True
                            break
                
                if not is_variant:
                    if ds_name in DATASET_CONFIGS:
                        csv_file = DATASET_CONFIGS[ds_name]['default_file']
                        test_csv_path = DATASET_CONFIGS[ds_name].get('test_file')
                        print(f"\n{'='*60}\nRunning experiment on dataset: {ds_name}\n{'='*60}")
                        print(f"File: {csv_file}")
                        ds_config = load_config('config_svm.json', dataset_name=ds_name)
                        run_acm_on_csv_file(csv_file, output_dir, config=ds_config, test_csv_path=test_csv_path)
        else:
            # Fallback to all configured dataset directories if nothing in config
            for ds_name in DATASET_CONFIGS:
                dataset_dir = DATASET_CONFIGS[ds_name].get('dir')
                if dataset_dir:
                    print(f"Running directory experiment on: {dataset_dir}")
                    ds_config = load_config('config_svm.json', dataset_name=ds_name)
                    run_acm_on_dataset_directory(dataset_dir, output_dir, config=ds_config)
