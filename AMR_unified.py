import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

try:
    from sklearn.impute import KNNImputer
except Exception:
    KNNImputer = None

try:
    import sklearn.experimental.enable_iterative_imputer
    from sklearn.impute import IterativeImputer
except Exception:
    IterativeImputer = None


DEFAULT_PARAMS = {
    "batch_size": 64,
    "lr_primal": 0.01,
    "lr_dual": 1.0,
    "c_reg": 1.0,
    "c_svm": 1.0,
    "fit_intercept": True,
    "select_k_samples": 10,
    "max_epochs": 50,
    "train_epochs_per_iter": 1,
    "loss_convergence_threshold": 1e-3,
    "repair_threshold_divisor": 100.0,
    "use_imputation_budget": True,
    "imputation_budget": 0.1,
    "knn_neighbors": 5,
    "mice_max_iter": 5,
    "mf_rank": 8,
    "mf_max_iter": 5,
    "sigmoid_clipping": 20.0,
    "residual_clipping": 2.0,
    "alpha_ema": 0.2,
    "random_state": 42,
    "test_size": 0.2,
}


SUPPORTED_MODELS = {"svm", "logistic", "linear"}
SUPPORTED_IMPUTERS = {"gt", "mean", "knn", "mice", "mf", "tcsdi", "llm"}


def _merge_params(params):
    merged = dict(DEFAULT_PARAMS)
    if params:
        merged.update(params)
    return merged


def nan_to_num_with_value(values, nan_value=0.0, posinf_value=1e6, neginf_value=-1e6):
    arr = np.asarray(values, dtype=float)
    try:
        return np.nan_to_num(arr, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
    except TypeError:
        arr = np.array(arr, copy=True)
        arr[np.isnan(arr)] = nan_value
        arr[np.isposinf(arr)] = posinf_value
        arr[np.isneginf(arr)] = neginf_value
        return arr


def _safe_array(values, fill=0.0):
    return nan_to_num_with_value(values, nan_value=fill, posinf_value=1e6, neginf_value=-1e6)


def normalize_classification_labels(y):
    labels = np.asarray(y, dtype=float).ravel()
    finite = labels[~np.isnan(labels)]
    unique = set(np.unique(finite))
    if unique == {0.0, 1.0}:
        return np.where(labels == 0.0, -1.0, 1.0)
    if unique == {2.0, 4.0}:
        return np.where(labels == 2.0, -1.0, 1.0)
    return np.where(labels > 0, 1.0, -1.0)


def regression_metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def load_csv_dataset(
    train_path,
    test_path=None,
    target_column=-1,
    has_header=True,
    model_type="svm",
    params=None,
):
    params = _merge_params(params)
    model_type = model_type.lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError("Unknown model_type: {}".format(model_type))

    train_df = pd.read_csv(train_path, header=0 if has_header else None)
    X, y = _extract_xy(train_df, target_column)
    if model_type in ("svm", "logistic"):
        y = normalize_classification_labels(y)

    if test_path:
        test_df = pd.read_csv(test_path, header=0 if has_header else None)
        X_test, y_test = _extract_xy(test_df, target_column)
        if model_type in ("svm", "logistic"):
            y_test = normalize_classification_labels(y_test)
        return X, y, X_test, y_test

    stratify = y if model_type in ("svm", "logistic") and len(np.unique(y)) > 1 else None
    return train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=stratify,
    )


def load_named_dataset(dataset_name, params=None):
    params = _merge_params(params)
    if dataset_name not in DATASET_CATALOG:
        raise KeyError("Unknown dataset entry: {}".format(dataset_name))
    entry = DATASET_CATALOG[dataset_name]
    model_type = entry.get("model_type", "svm")
    has_header = entry.get("has_header", True)
    target_column = entry.get("target_column", -1)

    train_df = pd.read_csv(entry["train_path"], header=0 if has_header else None)
    X, y = _extract_xy(train_df, target_column)
    if model_type in ("svm", "logistic"):
        y = normalize_classification_labels(y)

    X_gt = None
    X_train_gt = None
    gt_path = entry.get("gt_train_path")
    if gt_path:
        gt_df = pd.read_csv(gt_path, header=0 if entry.get("has_header", True) else None)
        X_gt, _ = _extract_xy(gt_df, target_column)

    if entry.get("test_path"):
        X_train, y_train = X, y
        test_df = pd.read_csv(entry["test_path"], header=0 if has_header else None)
        X_test, y_test = _extract_xy(test_df, target_column)
        if model_type in ("svm", "logistic"):
            y_test = normalize_classification_labels(y_test)
        X_train_gt = X_gt
    else:
        stratify = y if model_type in ("svm", "logistic") and len(np.unique(y)) > 1 else None
        indices = np.arange(len(y))
        X_train, X_test, y_train, y_test, train_idx, _ = train_test_split(
            X,
            y,
            indices,
            test_size=params["test_size"],
            random_state=params["random_state"],
            stratify=stratify,
        )
        if X_gt is not None:
            X_train_gt = X_gt[train_idx]

    return X_train, y_train, X_test, y_test, X_train_gt, entry


def _extract_xy(df, target_column):
    if isinstance(target_column, str):
        y = df[target_column].values.astype(float)
        X = df.drop(columns=[target_column]).values
    else:
        y = df.iloc[:, target_column].values.astype(float)
        X = df.drop(df.columns[target_column], axis=1).values
    X = pd.DataFrame(X).replace("", np.nan).values.astype(float)
    if np.isnan(y).any():
        raise ValueError("Target column contains NaN values.")
    return X, y


def prepare_arrays(X_train, y_train, X_test, y_test, model_type="svm", X_train_gt=None, params=None):
    params = _merge_params(params)
    model_type = model_type.lower()
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()

    if model_type in ("svm", "logistic"):
        y_train = normalize_classification_labels(y_train)
        y_test = normalize_classification_labels(y_test)

    train_mask = np.isnan(X_train)
    complete_rows = ~train_mask.any(axis=1)

    scaler = StandardScaler() if model_type == "linear" else MinMaxScaler()
    scaler_fit_matrix = X_train[complete_rows] if np.any(complete_rows) else SimpleImputer().fit_transform(X_train)
    scaler.fit(scaler_fit_matrix)

    col_min = nan_to_num_with_value(np.nanmin(X_train, axis=0), nan_value=0.0)
    col_max = nan_to_num_with_value(np.nanmax(X_train, axis=0), nan_value=1.0)
    scaled_min = scaler.transform(col_min.reshape(1, -1)).ravel()
    scaled_max = scaler.transform(col_max.reshape(1, -1)).ravel()
    lower = np.minimum(scaled_min, scaled_max)
    upper = np.maximum(scaled_min, scaled_max)

    n_samples, n_features = X_train.shape
    bounds = np.zeros((n_samples, n_features, 2), dtype=float)
    bounds[:, :, 0] = lower
    bounds[:, :, 1] = upper

    X_train_scaled = X_train.copy()
    observed = ~train_mask
    if np.any(observed):
        transformed = scaler.transform(nan_to_num_with_value(X_train, nan_value=0.0))
        X_train_scaled[observed] = transformed[observed]

    test_imputer = SimpleImputer(strategy="mean")
    test_reference = X_train[complete_rows] if np.any(complete_rows) else nan_to_num_with_value(X_train, nan_value=0.0)
    test_imputer.fit(test_reference)
    X_test_scaled = scaler.transform(test_imputer.transform(X_test))

    X_gt_scaled = None
    if X_train_gt is not None:
        X_gt_scaled = scaler.transform(np.asarray(X_train_gt, dtype=float))

    return X_train_scaled, y_train, X_test_scaled, y_test, train_mask, bounds, X_gt_scaled, scaler


class ImputationOracle:

    def impute(self, X_partial, mask, row_indices, method, context=None):
        raise NotImplementedError("External oracle must implement impute(...).")


class MatrixFactorizationImputer:

    def __init__(self, rank=8, max_iter=5):
        self.rank = int(rank)
        self.max_iter = int(max_iter)
        self.column_means = None
        self.reference = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.column_means = np.nanmean(X, axis=0)
        self.column_means = nan_to_num_with_value(self.column_means, nan_value=0.0)
        filled = np.where(np.isnan(X), self.column_means, X)
        observed = ~np.isnan(X)
        for _ in range(max(1, self.max_iter)):
            reconstructed = self._low_rank(filled)
            filled = np.where(observed, X, reconstructed)
        self.reference = filled
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.column_means is None:
            raise RuntimeError("MatrixFactorizationImputer must be fit before transform.")
        rows = np.where(np.isnan(X), self.column_means, X)
        observed = ~np.isnan(X)
        combined = np.vstack([self.reference, rows]) if self.reference is not None else rows
        for _ in range(max(1, self.max_iter)):
            reconstructed = self._low_rank(combined)
            row_part = reconstructed[-len(rows):]
            rows = np.where(observed, X, row_part)
            combined[-len(rows):] = rows
        return rows

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _low_rank(self, X):
        centered = X - np.mean(X, axis=0, keepdims=True)
        try:
            u, s, vt = np.linalg.svd(centered, full_matrices=False)
            rank = max(1, min(self.rank, len(s)))
            return (u[:, :rank] * s[:rank]).dot(vt[:rank]) + np.mean(X, axis=0, keepdims=True)
        except np.linalg.LinAlgError:
            return X


class BasicKNNImputer:

    def __init__(self, n_neighbors=5):
        self.n_neighbors = int(n_neighbors)
        self.reference = None
        self.column_means = None

    def fit(self, X):
        self.reference = np.asarray(X, dtype=float)
        self.column_means = nan_to_num_with_value(np.nanmean(self.reference, axis=0), nan_value=0.0)
        return self

    def transform(self, X):
        if self.reference is None:
            raise RuntimeError("BasicKNNImputer must be fit before transform.")
        X = np.asarray(X, dtype=float)
        filled = np.array(X, copy=True)
        ref = self.reference
        ref_observed = ~np.isnan(ref)

        for i in range(filled.shape[0]):
            missing_features = np.where(np.isnan(filled[i]))[0]
            if len(missing_features) == 0:
                continue
            observed_features = np.where(~np.isnan(filled[i]))[0]
            if len(observed_features) == 0:
                filled[i, missing_features] = self.column_means[missing_features]
                continue

            common_counts = np.sum(ref_observed[:, observed_features], axis=1)
            usable = common_counts > 0
            if not np.any(usable):
                filled[i, missing_features] = self.column_means[missing_features]
                continue

            diffs = ref[:, observed_features] - filled[i, observed_features]
            diffs = np.where(ref_observed[:, observed_features], diffs, 0.0)
            distances = np.sqrt(np.sum(diffs * diffs, axis=1) / np.maximum(common_counts, 1))
            distances[~usable] = np.inf
            neighbors = np.argsort(distances)[:max(1, self.n_neighbors)]

            for feature_idx in missing_features:
                neighbor_values = ref[neighbors, feature_idx]
                neighbor_values = neighbor_values[~np.isnan(neighbor_values)]
                if len(neighbor_values) == 0:
                    filled[i, feature_idx] = self.column_means[feature_idx]
                else:
                    filled[i, feature_idx] = float(np.mean(neighbor_values))
        return filled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def fit_imputer(method, X_train, params=None):
    params = _merge_params(params)
    method = method.lower()
    if method in ("gt", "tcsdi", "llm"):
        return None
    if method == "mean":
        return SimpleImputer(strategy="mean").fit(X_train)
    if method == "knn":
        if KNNImputer is None:
            return BasicKNNImputer(n_neighbors=params["knn_neighbors"]).fit(X_train)
        return KNNImputer(n_neighbors=params["knn_neighbors"]).fit(X_train)
    if method == "mice":
        if IterativeImputer is None:
            raise ImportError("MICE imputation requires IterativeImputer support.")
        return IterativeImputer(max_iter=params["mice_max_iter"], random_state=params["random_state"]).fit(X_train)
    if method == "mf":
        return MatrixFactorizationImputer(rank=params["mf_rank"], max_iter=params["mf_max_iter"]).fit(X_train)
    raise ValueError("Unknown imputation method: {}".format(method))


def impute_selected_rows(
    X_current,
    row_indices,
    mask,
    method,
    fitted_imputer=None,
    X_train_gt=None,
    oracle=None,
    context=None,
):
    method = method.lower()
    if method not in SUPPORTED_IMPUTERS:
        raise ValueError("Unknown imputation method: {}".format(method))
    if len(row_indices) == 0:
        return X_current

    X_new = X_current.copy()
    row_indices = np.asarray(row_indices, dtype=int)
    row_mask = mask[row_indices]

    if method == "gt":
        if X_train_gt is None:
            raise ValueError("GT imputation requires X_train_gt.")
        repaired = np.asarray(X_train_gt, dtype=float)[row_indices]
    elif method in ("tcsdi", "llm"):
        if oracle is None:
            raise RuntimeError("{} imputation requires an external oracle.".format(method.upper()))
        repaired = oracle.impute(
            X_partial=X_new[row_indices].copy(),
            mask=row_mask.copy(),
            row_indices=row_indices.copy(),
            method=method,
            context=context,
        )
    elif fitted_imputer is not None:
        repaired = fitted_imputer.transform(X_new[row_indices])
    else:
        repaired = SimpleImputer(strategy="mean").fit_transform(X_new)[row_indices]

    repaired = np.asarray(repaired, dtype=float)
    X_new[row_indices] = np.where(row_mask, repaired, X_new[row_indices])
    return X_new


class RobustSVMMinMax:
    def __init__(self, input_dim, n_samples, c_svm=1.0, lr_primal=0.01, lr_dual=1.0, fit_intercept=True):
        self.input_dim = input_dim
        self.n_samples = n_samples
        self.c_svm = c_svm
        self.lr_primal = lr_primal
        self.lr_dual = lr_dual
        self.fit_intercept = fit_intercept
        self.w = np.zeros(input_dim)
        self.b = 0.0
        self.alpha = np.full(n_samples, 0.5)
        self.lmbda = 1.0 / (n_samples * c_svm) if c_svm > 0 else 1e-4
        self.is_fitted = False

    def train_step(self, X_adv, y, indices):
        margins = y * (np.dot(X_adv, self.w) + self.b)
        grad_dual = 1.0 - _safe_array(margins)
        self.alpha[indices] = np.clip(self.alpha[indices] + self.lr_dual * np.clip(grad_dual, -1.0, 1.0), 0.0, 1.0)
        update_w = np.dot(self.alpha[indices] * y, X_adv) / max(1, len(indices))
        self.w = (1.0 - self.lr_primal * self.lmbda) * self.w + self.lr_primal * _safe_array(update_w)
        if self.fit_intercept:
            self.b += self.lr_primal * float(np.mean(self.alpha[indices] * y))
        self.w = _safe_array(self.w)
        self.b = float(_safe_array(self.b))
        self.is_fitted = True

    def adversarial_repair(self, X, bounds, mask, y):
        direction = y[:, np.newaxis] * self.w[np.newaxis, :]
        edge = np.where(direction > 0, bounds[:, :, 0], bounds[:, :, 1])
        return np.where(mask, edge, X)

    def compute_sensitivity(self, bounds, mask):
        width = bounds[:, :, 1] - bounds[:, :, 0]
        uncertainty = np.sum(np.abs(self.w) * width * mask, axis=1)
        return _safe_array((1.0 / self.n_samples) * self.alpha * uncertainty)

    def predict(self, X):
        scores = np.dot(_safe_array(X), self.w) + self.b
        return np.where(scores >= 0, 1.0, -1.0)

    def score(self, X, y):
        return float(accuracy_score(normalize_classification_labels(y), self.predict(X)))

    def diagnostics(self, X_train, y_train, bounds, mask):
        X_adv = self.adversarial_repair(X_train, bounds, mask, y_train)
        margins = y_train * (np.dot(X_adv, self.w) + self.b)
        primal = 0.5 * self.lmbda * np.dot(self.w, self.w) + np.mean(np.maximum(0.0, 1.0 - margins))
        lmbda_safe = max(self.lmbda, 1e-9)
        dual_grad_sum = np.dot(self.alpha * y_train, X_adv)
        dual = np.mean(self.alpha) - (1.0 / (2.0 * lmbda_safe * self.n_samples ** 2)) * np.dot(dual_grad_sum, dual_grad_sum)
        return float(primal), float(dual), float(primal - dual)


class RobustLogisticMinMax:
    def __init__(self, input_dim, n_samples, c_reg=1.0, lr_primal=0.01, lr_dual=1.0, fit_intercept=True, sigmoid_clipping=20.0):
        self.input_dim = input_dim
        self.n_samples = n_samples
        self.c_reg = c_reg
        self.lr_primal = lr_primal
        self.lr_dual = lr_dual
        self.fit_intercept = fit_intercept
        self.sigmoid_clipping = sigmoid_clipping
        self.w = np.random.normal(0.0, 0.01, input_dim)
        self.b = 0.0
        self.alpha = np.full(n_samples, 0.5)
        self.lmbda = 1.0 / (n_samples * c_reg) if c_reg > 0 else 1e-4
        self.is_fitted = False

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -self.sigmoid_clipping, self.sigmoid_clipping)))

    def train_step(self, X_adv, y, indices):
        logits = np.dot(X_adv, self.w) + self.b
        margins = y * logits
        self.alpha[indices] = self._sigmoid(-margins)
        update_w = np.dot(self.alpha[indices] * y, X_adv) / max(1, len(indices))
        self.w = (1.0 - self.lr_primal * self.lmbda) * self.w + self.lr_primal * _safe_array(update_w)
        if self.fit_intercept:
            self.b += self.lr_primal * float(np.mean(self.alpha[indices] * y))
        self.w = _safe_array(self.w)
        self.b = float(_safe_array(self.b))
        self.is_fitted = True

    def adversarial_repair(self, X, bounds, mask, y):
        direction = y[:, np.newaxis] * self.w[np.newaxis, :]
        edge = np.where(direction > 0, bounds[:, :, 0], bounds[:, :, 1])
        return np.where(mask, edge, X)

    def compute_sensitivity(self, bounds, mask):
        width = bounds[:, :, 1] - bounds[:, :, 0]
        uncertainty = np.sum(np.abs(self.w) * width * mask, axis=1)
        w_norm = np.linalg.norm(self.w) + 1e-9
        return _safe_array((1.0 / self.n_samples) * self.alpha * (uncertainty / w_norm))

    def predict(self, X):
        scores = np.dot(_safe_array(X), self.w) + self.b
        return np.where(scores >= 0, 1.0, -1.0)

    def score(self, X, y):
        return float(accuracy_score(normalize_classification_labels(y), self.predict(X)))

    def diagnostics(self, X_train, y_train, bounds, mask):
        X_adv = self.adversarial_repair(X_train, bounds, mask, y_train)
        margins = y_train * (np.dot(X_adv, self.w) + self.b)
        log_losses = np.log1p(np.exp(-np.clip(margins, -self.sigmoid_clipping, self.sigmoid_clipping)))
        primal = 0.5 * self.lmbda * np.dot(self.w, self.w) + np.mean(log_losses)
        eps = 1e-9
        p = np.clip(self.alpha, eps, 1.0 - eps)
        entropy = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
        lmbda_safe = max(self.lmbda, 1e-9)
        dual_grad_sum = np.dot(self.alpha * y_train, X_adv)
        dual = np.mean(entropy) - (1.0 / (2.0 * lmbda_safe * self.n_samples ** 2)) * np.dot(dual_grad_sum, dual_grad_sum)
        return float(primal), float(dual), float(primal - dual)


class RobustLinearRegressionMinMax:
    def __init__(self, input_dim, n_samples, c_reg=1.0, lr_primal=0.01, lr_dual=1.0, fit_intercept=True, residual_clipping=2.0, alpha_ema=0.2):
        self.input_dim = input_dim
        self.n_samples = n_samples
        self.c_reg = c_reg
        self.lr_primal = lr_primal
        self.lr_dual = lr_dual
        self.fit_intercept = fit_intercept
        self.residual_clipping = residual_clipping
        self.alpha_ema = alpha_ema
        self.w = np.zeros(input_dim)
        self.b = 0.0
        self.alpha = np.full(n_samples, 0.5)
        self.lmbda = 1.0 / (n_samples * c_reg) if c_reg > 0 else 1e-4
        self.is_fitted = False

    def train_step(self, X_adv, y, indices):
        predictions = np.dot(X_adv, self.w) + self.b
        residuals = _safe_array(predictions - y)
        clipped = np.clip(residuals, -self.residual_clipping, self.residual_clipping)
        scale = max(float(self.residual_clipping), 1e-12)
        target_alpha = np.clip(np.abs(clipped) / scale, 0.0, 1.0)
        eta = np.clip(float(self.alpha_ema) * float(self.lr_dual), 0.0, 1.0)
        self.alpha[indices] = (1.0 - eta) * self.alpha[indices] + eta * target_alpha
        update_w = np.dot(self.alpha[indices] * clipped, X_adv) / max(1, len(indices))
        self.w = (1.0 - self.lr_primal * self.lmbda) * self.w - self.lr_primal * _safe_array(update_w)
        if self.fit_intercept:
            self.b -= self.lr_primal * float(np.mean(self.alpha[indices] * clipped))
        self.w = _safe_array(self.w)
        self.b = float(_safe_array(self.b))
        self.is_fitted = True

    def adversarial_repair(self, X, bounds, mask, y):
        lower = bounds[:, :, 0]
        upper = bounds[:, :, 1]
        min_corner = np.where(self.w[np.newaxis, :] >= 0, lower, upper)
        max_corner = np.where(self.w[np.newaxis, :] >= 0, upper, lower)
        X_min = np.where(mask, min_corner, X)
        X_max = np.where(mask, max_corner, X)
        pred_min = np.dot(_safe_array(X_min), self.w) + self.b
        pred_max = np.dot(_safe_array(X_max), self.w) + self.b
        choose_max = np.abs(y - pred_max) >= np.abs(y - pred_min)
        return np.where(choose_max[:, np.newaxis], X_max, X_min)

    def compute_sensitivity(self, bounds, mask):
        width = bounds[:, :, 1] - bounds[:, :, 0]
        uncertainty = np.sum(np.abs(self.w) * width * mask, axis=1)
        return _safe_array((1.0 / self.n_samples) * self.alpha * uncertainty)

    def predict(self, X):
        return np.dot(_safe_array(X), self.w) + self.b

    def score(self, X, y):
        return -float(mean_squared_error(y, self.predict(X)))

    def diagnostics(self, X_train, y_train, bounds, mask):
        X_adv = self.adversarial_repair(X_train, bounds, mask, y_train)
        predictions = np.dot(_safe_array(X_adv), self.w) + self.b
        losses = 0.5 * (y_train - predictions) ** 2
        primal = 0.5 * self.lmbda * np.dot(self.w, self.w) + np.mean(losses)
        lmbda_safe = max(self.lmbda, 1e-9)
        dual_grad_sum = np.dot(self.alpha * (y_train - predictions), X_adv)
        surrogate = np.mean(self.alpha) - (1.0 / (2.0 * lmbda_safe * self.n_samples ** 2)) * np.dot(dual_grad_sum, dual_grad_sum)
        return float(primal), float(surrogate), float(primal - surrogate)


def make_model(model_type, input_dim, n_samples, params=None):
    params = _merge_params(params)
    model_type = model_type.lower()
    if model_type == "svm":
        return RobustSVMMinMax(
            input_dim=input_dim,
            n_samples=n_samples,
            c_svm=params["c_svm"],
            lr_primal=params["lr_primal"],
            lr_dual=params["lr_dual"],
            fit_intercept=params["fit_intercept"],
        )
    if model_type == "logistic":
        return RobustLogisticMinMax(
            input_dim=input_dim,
            n_samples=n_samples,
            c_reg=params["c_reg"],
            lr_primal=params["lr_primal"],
            lr_dual=params["lr_dual"],
            fit_intercept=params["fit_intercept"],
            sigmoid_clipping=params["sigmoid_clipping"],
        )
    if model_type == "linear":
        return RobustLinearRegressionMinMax(
            input_dim=input_dim,
            n_samples=n_samples,
            c_reg=params["c_reg"],
            lr_primal=params["lr_primal"],
            lr_dual=params["lr_dual"],
            fit_intercept=params["fit_intercept"],
            residual_clipping=params["residual_clipping"],
            alpha_ema=params["alpha_ema"],
        )
    raise ValueError("Unknown model_type: {}".format(model_type))


def warm_start_model(model, model_type, X_current, y_train, mask, params=None):
    params = _merge_params(params)
    complete_rows = ~mask.any(axis=1)
    clean_indices = np.where(complete_rows)[0]
    if len(clean_indices) == 0:
        return model

    X_clean = X_current[clean_indices]
    y_clean = y_train[clean_indices]
    if model_type == "svm" and len(np.unique(y_clean)) > 1:
        alpha = 1.0 / (len(y_train) * params["c_svm"]) if params["c_svm"] > 0 else 1e-4
        clf = SGDClassifier(
            loss="hinge",
            alpha=alpha,
            fit_intercept=params["fit_intercept"],
            max_iter=20,
            random_state=params["random_state"],
        )
        clf.fit(X_clean, y_clean)
        model.w = _safe_array(clf.coef_[0])
        model.b = float(clf.intercept_[0]) if params["fit_intercept"] else 0.0
        model.is_fitted = True
    elif model_type == "logistic" and len(np.unique(y_clean)) > 1:
        clf = LogisticRegression(
            C=params["c_reg"],
            fit_intercept=params["fit_intercept"],
            max_iter=200,
            random_state=params["random_state"],
        )
        clf.fit(X_clean, y_clean)
        model.w = _safe_array(clf.coef_.ravel())
        model.b = float(clf.intercept_[0]) if params["fit_intercept"] else 0.0
        logits = np.dot(np.where(np.isnan(X_current), 0.5, X_current), model.w) + model.b
        model.alpha = 1.0 / (1.0 + np.exp(-np.clip(y_train * logits, -20, 20)))
        model.is_fitted = True
    elif model_type == "linear":
        ridge = Ridge(alpha=1.0 / params["c_reg"] if params["c_reg"] > 0 else 1.0, fit_intercept=params["fit_intercept"])
        ridge.fit(X_clean, y_clean)
        model.w = _safe_array(ridge.coef_)
        model.b = float(ridge.intercept_) if params["fit_intercept"] else 0.0
        residuals = model.predict(X_clean) - y_clean
        scale = max(float(params["residual_clipping"]), 1e-12)
        model.alpha[clean_indices] = np.clip(np.abs(np.clip(residuals, -scale, scale)) / scale, 0.0, 1.0)
        model.is_fitted = True
    return model


def run_impute_all_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    model_type="svm",
    imputation_method="mean",
    X_train_gt=None,
    params=None,
    imputation_oracle=None,
):
    start = time.time()
    params = _merge_params(params)
    X_train_p, y_train_p, X_test_p, y_test_p, _mask, _bounds, X_gt_p, _scaler = prepare_arrays(
        X_train, y_train, X_test, y_test, model_type=model_type, X_train_gt=X_train_gt, params=params
    )
    all_rows = np.where(np.isnan(X_train_p).any(axis=1))[0]
    current_mask = np.isnan(X_train_p)
    imputer = fit_imputer(imputation_method, X_train_p, params=params)
    X_filled = impute_selected_rows(
        X_train_p,
        all_rows,
        current_mask,
        imputation_method,
        fitted_imputer=imputer,
        X_train_gt=X_gt_p,
        oracle=imputation_oracle,
    )
    X_filled = SimpleImputer(strategy="mean").fit_transform(X_filled)

    if model_type == "svm":
        alpha = 1.0 / (len(y_train_p) * params["c_svm"]) if params["c_svm"] > 0 else 1e-4
        model = SGDClassifier(loss="hinge", alpha=alpha, fit_intercept=params["fit_intercept"], random_state=params["random_state"])
        model.fit(X_filled, y_train_p)
        metric = {"accuracy": float(model.score(X_test_p, y_test_p))}
    elif model_type == "logistic":
        model = LogisticRegression(C=params["c_reg"], fit_intercept=params["fit_intercept"], max_iter=1000)
        model.fit(X_filled, y_train_p)
        metric = {"accuracy": float(model.score(X_test_p, y_test_p))}
    else:
        model = Ridge(alpha=1.0 / params["c_reg"] if params["c_reg"] > 0 else 1.0, fit_intercept=params["fit_intercept"])
        model.fit(X_filled, y_train_p)
        metric = regression_metrics(y_test_p, model.predict(X_test_p))

    metric.update({
        "total_time": time.time() - start,
        "total_imputed_count": int(len(all_rows)),
        "imputation_ratio": 1.0 if len(all_rows) else 0.0,
    })
    return metric


def run_amr(
    X_train,
    y_train,
    X_test,
    y_test,
    model_type="svm",
    imputation_method="knn",
    X_train_gt=None,
    params=None,
    imputation_oracle=None,
):
    params = _merge_params(params)
    model_type = model_type.lower()
    imputation_method = imputation_method.lower()
    if model_type not in SUPPORTED_MODELS:
        raise ValueError("Unsupported model_type: {}".format(model_type))
    if imputation_method not in SUPPORTED_IMPUTERS:
        raise ValueError("Unsupported imputation_method: {}".format(imputation_method))
    if params["random_state"] is not None:
        np.random.seed(int(params["random_state"]))

    X_current, y_train, X_test_scaled, y_test, current_mask, current_bounds, X_gt_scaled, _scaler = prepare_arrays(
        X_train, y_train, X_test, y_test, model_type=model_type, X_train_gt=X_train_gt, params=params
    )
    n_samples, n_features = X_current.shape
    num_incomplete = int(np.sum(current_mask.any(axis=1)))
    incomplete_rows = current_mask.any(axis=1)
    already_imputed = np.zeros(n_samples, dtype=bool)
    imputed_indices = []
    total_imputed = 0
    max_allowed = int(params["imputation_budget"] * num_incomplete) if params["use_imputation_budget"] else None

    start = time.time()
    model = make_model(model_type, n_features, n_samples, params=params)
    model = warm_start_model(model, model_type, X_current, y_train, current_mask, params=params)
    fitted_imputer = fit_imputer(imputation_method, X_current, params=params)

    final_epoch = 0
    final_sensitivity = np.zeros(n_samples)
    history = []

    for epoch in range(int(params["max_epochs"])):
        final_sensitivity = model.compute_sensitivity(current_bounds, current_mask)
        sum_sensitivity = float(np.sum(final_sensitivity))

        candidates = np.where((~already_imputed) & incomplete_rows)[0]
        imputed_this_epoch = 0
        if len(candidates) > 0:
            candidate_scores = final_sensitivity[candidates]
            threshold = float(params["loss_convergence_threshold"]) / max(float(params["repair_threshold_divisor"]), 1e-12)
            valid = candidate_scores > threshold
            if np.any(valid):
                valid_candidates = candidates[valid]
                valid_scores = candidate_scores[valid]
                num_to_repair = min(len(valid_candidates), int(params["select_k_samples"]))
                if max_allowed is not None:
                    num_to_repair = min(num_to_repair, max(0, max_allowed - total_imputed))

                if num_to_repair > 0:
                    chosen = valid_candidates[np.argsort(-valid_scores)[:num_to_repair]]
                    X_current = impute_selected_rows(
                        X_current,
                        chosen,
                        current_mask,
                        imputation_method,
                        fitted_imputer=fitted_imputer,
                        X_train_gt=X_gt_scaled,
                        oracle=imputation_oracle,
                        context={"model_type": model_type, "params": params},
                    )
                    repaired = X_current[chosen]
                    current_bounds[chosen, :, 0] = repaired
                    current_bounds[chosen, :, 1] = repaired
                    current_mask[chosen, :] = False
                    incomplete_rows[chosen] = False
                    already_imputed[chosen] = True
                    imputed_indices.extend(int(i) for i in chosen)
                    total_imputed += len(chosen)
                    imputed_this_epoch = len(chosen)

        for _ in range(int(params["train_epochs_per_iter"])):
            for batch_idx in _batch_indices(n_samples, int(params["batch_size"])):
                X_batch = X_current[batch_idx]
                y_batch = y_train[batch_idx]
                bounds_batch = current_bounds[batch_idx]
                mask_batch = current_mask[batch_idx]
                X_adv = model.adversarial_repair(X_batch, bounds_batch, mask_batch, y_batch)
                model.train_step(_safe_array(X_adv), y_batch, batch_idx)

        primal, dual_or_surrogate, gap = model.diagnostics(X_current, y_train, current_bounds, current_mask)
        metrics = _evaluate_model(model, model_type, X_test_scaled, y_test)
        final_epoch = epoch + 1
        epoch_record = {
            "epoch": final_epoch,
            "sum_sensitivity": sum_sensitivity,
            "imputed_this_epoch": imputed_this_epoch,
            "primal": primal,
            "dual_or_surrogate": dual_or_surrogate,
            "gap": gap,
        }
        epoch_record.update(metrics)
        history.append(epoch_record)

        if sum_sensitivity < float(params["loss_convergence_threshold"]):
            break

    final_metrics = _evaluate_model(model, model_type, X_test_scaled, y_test)
    return {
        "model_type": model_type,
        "imputation_method": imputation_method,
        "total_time": time.time() - start,
        "total_imputed_count": int(total_imputed),
        "imputation_ratio": float(total_imputed / num_incomplete) if num_incomplete else 0.0,
        "num_incomplete_train": int(num_incomplete),
        "total_epochs": int(final_epoch),
        "uncertainty_range": float(np.mean(final_sensitivity)) if len(final_sensitivity) else 0.0,
        "imputed_indices": imputed_indices,
        "metrics": final_metrics,
        "history": history,
    }


def _batch_indices(n_samples, batch_size):
    indices = np.random.permutation(n_samples)
    for start in range(0, n_samples, max(1, batch_size)):
        yield indices[start:start + max(1, batch_size)]


def _evaluate_model(model, model_type, X_test, y_test):
    if model_type in ("svm", "logistic"):
        return {"accuracy": model.score(X_test, y_test)}
    return regression_metrics(y_test, model.predict(X_test))


if __name__ == "__main__":
    dataset_name = "heart_MCAR_20"
    params = dict(DEFAULT_PARAMS)
    params.update({
        "max_epochs": 50,
        "select_k_samples": 10,
        "imputation_budget": 0.1,
        "random_state": 42,
    })

    try:
        X_train, y_train, X_test, y_test, X_train_gt, dataset_info = load_named_dataset(dataset_name, params=params)
        result = run_amr(
            X_train,
            y_train,
            X_test,
            y_test,
            model_type=dataset_info.get("model_type", "svm"),
            imputation_method="knn",
            X_train_gt=X_train_gt,
            params=params,
        )
        print("AMR completed for {}".format(dataset_name))
        print("Model: {}".format(result["model_type"]))
        print("Imputation method: {}".format(result["imputation_method"]))
        print("Metrics: {}".format(result["metrics"]))
        print("Imputation ratio: {:.4f}".format(result["imputation_ratio"]))
    except FileNotFoundError:
        print("Dataset files were not found. Update DATASET_CATALOG with local CSV paths before running.")
