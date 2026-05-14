
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import rtdl_revisiting_models as rtdl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import os
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
import csv
import argparse
import json
import math

from Imputation_method import knn_imputation, mice_impute, rf_iterative_impute

def load_config(config_path, dataset_name=None):
    with open(config_path, 'r') as f:
        config = json.load(f)
    hypers = config.get('default', {}).copy()
    if dataset_name and dataset_name in config:
        hypers.update(config[dataset_name])
    if 'global_settings' in config:
        hypers['global_settings'] = config['global_settings']
    return hypers

def generate_random_seeds(num_seeds=3, seed_range=(1, 10000)):
    import random
    random.seed(int(time.time()))
    seeds = []
    while len(seeds) < num_seeds:
        s = random.randint(seed_range[0], seed_range[1])
        if s not in seeds:
            seeds.append(s)
    return seeds

def create_lr_scheduler(optimizer, total_steps, warmup_fraction=0.1):
    warmup_steps = int(total_steps * warmup_fraction)
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return max(1e-2, current_step / max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class RobustFTTransformer:

    def __init__(self, n_features: int, config: dict):
        n_blocks      = config.get('N_BLOCKS', 2)
        d_block       = config.get('D_BLOCK', 64)
        attn_heads    = config.get('ATTENTION_N_HEADS', 8)
        attn_drop     = config.get('ATTENTION_DROPOUT', 0.1)
        ffn_mult      = config.get('FFN_D_HIDDEN_MULTIPLIER', 4/3)
        ffn_drop      = config.get('FFN_DROPOUT', 0.0)
        res_drop      = config.get('RESIDUAL_DROPOUT', 0.0)
        lr            = config.get('STEP_SIZE', 1e-4)
        weight_decay  = config.get('WEIGHT_DECAY', 1e-5)

        self.lambda_reg    = config.get('LAMBDA_REG', 0.01)
        self.grad_clip_norm = config.get('GRAD_CLIP_NORM', 1.0)
        self.n_features    = n_features
        self.predict_batch_size = config.get('PREDICT_BATCH_SIZE', 8192)
        use_cuda = config.get('USE_CUDA', True)
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        if d_block % attn_heads != 0:
            d_block = ((d_block // attn_heads) + 1) * attn_heads

        self.model = rtdl.FTTransformer(
            n_cont_features=n_features,
            cat_cardinalities=[],
            d_out=1,
            n_blocks=n_blocks,
            d_block=d_block,
            attention_n_heads=attn_heads,
            attention_dropout=attn_drop,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=ffn_mult,
            ffn_dropout=ffn_drop,
            residual_dropout=res_drop,
        ).to(self.device)

        param_groups = self.model.make_parameter_groups()
        for g in param_groups:
            if 'weight_decay' not in g:
                g['weight_decay'] = weight_decay
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr)

    def compute_hsas(self, x_l: np.ndarray, x_u: np.ndarray,
                     y_labels: np.ndarray, batch_size: int = 128) -> np.ndarray:
        self.model.eval()
        all_hsas = []

        with torch.enable_grad():
            for i in range(0, len(x_l), batch_size):
                x_l_b = torch.tensor(x_l[i:i + batch_size], dtype=torch.float32,
                                     device=self.device)
                x_u_b = torch.tensor(x_u[i:i + batch_size], dtype=torch.float32,
                                     device=self.device)
                y_b   = torch.tensor(
                    (y_labels[i:i + batch_size] + 1) / 2.0, dtype=torch.float32,
                    device=self.device
                )

                diam = (x_u_b - x_l_b)

                x_center = (x_l_b + x_u_b) / 2.0

                embeddings = self.model.cont_embeddings(x_center)
                cls_tok    = self.model.cls_embedding(x_center.shape[:-1])
                x_tokens   = torch.cat([cls_tok, embeddings], dim=1)
                out        = self.model.backbone(x_tokens).squeeze(-1)
                loss       = F.binary_cross_entropy_with_logits(out, y_b, reduction='none')

                grad_emb = torch.autograd.grad(
                    loss.sum(), embeddings, create_graph=False
                )[0]

                grad_emb_sq = (grad_emb ** 2).sum(dim=-1)
                diam_sq     = diam ** 2
                hsas_sq     = (grad_emb_sq * diam_sq).sum(dim=1)
                hsas        = torch.sqrt(hsas_sq.clamp(min=0))

                all_hsas.append(hsas.detach().cpu().numpy())

        return np.concatenate(all_hsas) if all_hsas else np.array([])

    def train_step(self, x_l: torch.Tensor, x_u: torch.Tensor,
                   y: torch.Tensor, is_robust: bool = True,
                   pos_weight: torch.Tensor = None) -> float:
        x_l = x_l.to(self.device, non_blocking=True)
        x_u = x_u.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device, non_blocking=True)

        y_target = (y + 1.0) / 2.0
        x_c = (x_l + x_u) / 2.0

        if is_robust:
            self.model.eval()
            with torch.enable_grad():
                x_adv_base  = x_c.detach().requires_grad_(True)
                out_adv_dir = self.model(x_cont=x_adv_base, x_cat=None).squeeze(-1)
                loss_adv_dir = F.binary_cross_entropy_with_logits(
                    out_adv_dir, y_target, pos_weight=pos_weight)
                grad_dir = torch.autograd.grad(
                    loss_adv_dir, x_adv_base, create_graph=False
                )[0].detach()

            half_diam = (x_u - x_l) / 2.0
            x_adv = (x_c.detach() + half_diam * torch.sign(grad_dir)).detach()
            x_adv = torch.max(x_adv, x_l)
            x_adv = torch.min(x_adv, x_u)

            self.model.train()
            self.optimizer.zero_grad()
            out_nom = self.model(x_cont=x_c.detach(), x_cat=None).squeeze(-1)
            loss_nom = F.binary_cross_entropy_with_logits(
                out_nom, y_target, pos_weight=pos_weight)

            out_adv_f = self.model(x_cont=x_adv, x_cat=None).squeeze(-1)
            loss_adv_f = F.binary_cross_entropy_with_logits(
                out_adv_f, y_target, pos_weight=pos_weight)

            total_loss = loss_nom + self.lambda_reg * loss_adv_f
            total_loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip_norm)
            self.optimizer.step()
            return loss_nom.item()

        else:
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(x_cont=x_c.detach(), x_cat=None).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                out, y_target, pos_weight=pos_weight)
            loss.backward()
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.grad_clip_norm)
            self.optimizer.step()
            return loss.item()

    def predict(self, X: np.ndarray, batch_size: int = None) -> np.ndarray:
        self.model.eval()
        if batch_size is None:
            batch_size = self.predict_batch_size
        preds = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                X_t = torch.tensor(X[start:start + batch_size],
                                   dtype=torch.float32, device=self.device)
                out = self.model(x_cont=X_t, x_cat=None).squeeze(-1)
                pred = torch.where(out >= 0, 1.0, -1.0)
                preds.append(pred.detach().cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        y_true = np.where(y > 0, 1.0, -1.0)
        return accuracy_score(y_true, y_pred)

    def predict_distribution(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        y_true = np.where(y > 0, 1.0, -1.0)
        acc = float(accuracy_score(y_true, y_pred))
        n_pos = int(np.sum(y_pred == 1.0))
        n_neg = int(np.sum(y_pred == -1.0))
        tp = int(np.sum((y_pred == 1.0) & (y_true == 1.0)))
        tn = int(np.sum((y_pred == -1.0) & (y_true == -1.0)))
        fp = int(np.sum((y_pred == 1.0) & (y_true == -1.0)))
        fn = int(np.sum((y_pred == -1.0) & (y_true == 1.0)))
        return {"acc": acc, "pred_pos": n_pos, "pred_neg": n_neg,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}

def load_dataset_from_csv(csv_file_path, config=None, test_csv_path=None):
    test_size    = config.get('TEST_SIZE', 0.2) if config else 0.2
    random_state = config.get('RANDOM_STATE', 42) if config else 42

    dataset_name = os.path.basename(csv_file_path).replace(".csv", "")
    df = pd.read_csv(csv_file_path, header='infer')

    if "breast" in dataset_name.lower():
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values.astype(np.float32)
        unique_labels = np.unique(y[~np.isnan(y)])
        if set(unique_labels) == {2.0, 4.0}:
            y = np.where(y == 2.0, -1.0, 1.0)
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values.astype(np.float32)

    unique_labels = np.unique(y[~np.isnan(y)])
    if set(unique_labels) == {0.0, 1.0}:
        y = np.where(y == 0.0, -1.0, 1.0)

    X = pd.DataFrame(X).replace("", np.nan).values.astype(np.float32)

    X_train_gt = None
    _data_root = os.environ.get("DATA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    base_ds_map = {
        'malware':        os.path.join(_data_root, 'malware_OG_train.csv'),
        'tuadromd':       os.path.join(_data_root, 'tuadromd_OG_train.csv'),
        'default':        os.path.join(_data_root, 'default_OG_train.csv'),
        'credit_default': os.path.join(_data_root, 'default_OG_train.csv'),
        'fraud':          os.path.join(_data_root, 'fraud_OG_train.csv'),
        'higgs':          os.path.join(os.environ.get("HIGGS_DATA_DIR", "/path/to/higgs/data"), 'higgs_OG_train.csv'),
    }
    og_file_path = None
    for key, path in base_ds_map.items():
        if dataset_name.startswith(key):
            og_file_path = path
            break
    if og_file_path and os.path.exists(og_file_path):
        try:
            df_og = pd.read_csv(og_file_path, header='infer')
            if "breast" in dataset_name.lower():
                X_gt_raw = df_og.iloc[:, 1:-1].values
            else:
                X_gt_raw = df_og.iloc[:, :-1].values
            X_train_gt = pd.DataFrame(X_gt_raw).replace("", np.nan).values.astype(np.float32)
            print(f"Loaded Ground Truth (OG) training set from: {og_file_path}")
        except Exception as e:
            print(f"Warning: Could not load ground truth file {og_file_path}: {e}")

    if test_csv_path and os.path.exists(test_csv_path):
        X_train = X
        y_train = y
        df_test = pd.read_csv(test_csv_path, header='infer')
        if "breast" in dataset_name.lower():
            X_test = df_test.iloc[:, 1:-1].values
            y_test = df_test.iloc[:, -1].values.astype(np.float32)
        else:
            X_test = df_test.iloc[:, :-1].values
            y_test = df_test.iloc[:, -1].values.astype(np.float32)
        unique_test = np.unique(y_test[~np.isnan(y_test)])
        if set(unique_test) == {0.0, 1.0}:
            y_test = np.where(y_test == 0.0, -1.0, 1.0)
        elif set(unique_test) == {2.0, 4.0}:
            y_test = np.where(y_test == 2.0, -1.0, 1.0)
        X_test = pd.DataFrame(X_test).replace("", np.nan).values.astype(np.float32)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        if X_train_gt is not None:
            n_align = min(X_train_gt.shape[0], X.shape[0])
            X_gt_aligned = X_train_gt[:n_align]
            y_align = y[:n_align]
            X_train_gt, _, _, _ = train_test_split(
                X_gt_aligned, y_align, test_size=test_size, random_state=random_state,
                stratify=y_align if len(np.unique(y_align)) > 1 else None
            )

    mask_train = np.isnan(X_train)
    col_min = np.nan_to_num(np.nanmin(X_train, axis=0), nan=0.0)
    col_max = np.nan_to_num(np.nanmax(X_train, axis=0), nan=1.0)

    scaler = MinMaxScaler()
    complete_rows = ~np.isnan(X_train).any(axis=1)
    if np.sum(complete_rows) > 0:
        scaler.fit(X_train[complete_rows])
    else:
        temp_imp = SimpleImputer(strategy="mean")
        scaler.fit(temp_imp.fit_transform(X_train))

    n_train, n_features = X_train.shape
    bounds_train = np.zeros((n_train, n_features, 2), dtype=np.float32)

    X_train_filled = np.nan_to_num(X_train, nan=0.0)
    X_train_scaled_all = scaler.transform(X_train_filled)
    X_train_scaled = X_train.copy()
    non_nan_mask = ~np.isnan(X_train)
    X_train_scaled[non_nan_mask] = X_train_scaled_all[non_nan_mask]

    col_min_scaled = scaler.transform(col_min.reshape(1, -1)).flatten()
    col_max_scaled = scaler.transform(col_max.reshape(1, -1)).flatten()

    bounds_train[:, :, 0] = np.where(mask_train, col_min_scaled[None, :], X_train_scaled)
    bounds_train[:, :, 1] = np.where(mask_train, col_max_scaled[None, :], X_train_scaled)

    if X_train_gt is not None:
        X_train_gt = scaler.transform(np.nan_to_num(X_train_gt, nan=0.0)).astype(np.float32, copy=False)

    X_train_scaled = X_train_scaled.astype(np.float32, copy=False)
    bounds_train = bounds_train.astype(np.float32, copy=False)
    X_test_scaled = scaler.transform(np.nan_to_num(X_test, nan=0.0)).astype(np.float32, copy=False)
    num_incomplete_train = int(np.sum(np.any(mask_train, axis=1)))
    return (X_train_scaled, y_train, X_test_scaled, y_test,
            dataset_name, mask_train, bounds_train, num_incomplete_train, X_train_gt)

def impute_train_test_for_ft(X_train_inc, X_test, method="mean", config=None, X_train_gt=None):
    config = config or {}
    method = (method or "mean").lower()
    X_train_inc = np.asarray(X_train_inc, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    if method == "gt" and X_train_gt is not None:
        X_tr = np.asarray(X_train_gt, dtype=np.float32)
        return X_tr, np.nan_to_num(X_test, nan=0.0).astype(np.float32, copy=False)

    if method == "knn":
        if KNNImputer is None:
            raise ImportError("KNNImputer is unavailable in this scikit-learn installation.")
        imputer = KNNImputer(n_neighbors=config.get("KNN_NEIGHBORS", 5))
        X_tr = imputer.fit_transform(X_train_inc)
        X_te = imputer.transform(X_test)
        return (
            np.asarray(X_tr, dtype=np.float32),
            np.asarray(X_te, dtype=np.float32),
        )

    if method == "mice":
        max_iter = config.get("MICE_MAX_ITER", 2)
        random_state = config.get("RANDOM_STATE", 42)
        imputer, _ = mice_impute(
            X_train_inc,
            max_iter=max_iter,
            random_state=random_state,
            condition="Fit",
            imputer=None,
        )
        X_tr, _ = mice_impute(
            X_train_inc,
            max_iter=max_iter,
            random_state=random_state,
            condition="Transform",
            imputer=imputer,
        )
        X_te, _ = mice_impute(
            X_test,
            max_iter=max_iter,
            random_state=random_state,
            condition="Transform",
            imputer=imputer,
        )
        return (
            np.asarray(X_tr, dtype=np.float32),
            np.asarray(X_te, dtype=np.float32),
        )

    imputer = SimpleImputer(strategy="mean")
    X_tr = imputer.fit_transform(X_train_inc)
    X_te = imputer.transform(X_test)
    return (
        np.asarray(X_tr, dtype=np.float32),
        np.asarray(X_te, dtype=np.float32),
    )

def run_ft_baseline_impute_all(X_train_inc, y_train, X_test, y_test,
                               method="gt", config=None, X_train_gt=None,
                               seed=42, log_path=None, log_every=5):
    start_time = time.time()
    method = (method or "mean").lower()

    X_tr, X_te = impute_train_test_for_ft(
        X_train_inc, X_test, method=method, config=config, X_train_gt=X_train_gt
    )
    y_tr = np.array(y_train, dtype=np.float32)
    if X_tr.shape[0] != y_tr.shape[0]:
        mn = min(X_tr.shape[0], y_tr.shape[0])
        print(
            f"  [Baseline] length mismatch after {method} imputation "
            f"(X={X_tr.shape[0]}, y={y_tr.shape[0]}). Truncating both to {mn}.",
            flush=True,
        )
        X_tr, y_tr = X_tr[:mn], y_tr[:mn]

    X_tr = np.nan_to_num(X_tr, nan=0.0).astype(np.float32, copy=False)
    X_te = np.nan_to_num(X_te, nan=0.0).astype(np.float32, copy=False)

    n_features = X_tr.shape[1]
    y_np = np.where(np.array(y_tr, dtype=np.float32).ravel() > 0, 1.0, -1.0)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    model = RobustFTTransformer(n_features=n_features, config=config)

    max_epochs  = config.get('MAX_TOTAL_EPOCHS', 50) if config else 50
    train_eps   = config.get('TRAIN_EPOCHS_PER_ITER', 10) if config else 10
    batch_size  = config.get('BATCH_SIZE', 64) if config else 64
    total_eps   = max_epochs * train_eps
    n_samples   = X_tr.shape[0]
    indices     = np.arange(n_samples)

    use_class_weight = config.get('USE_CLASS_WEIGHT', True) if config else True
    cw_power = config.get('CLASS_WEIGHT_POWER', 0.5) if config else 0.5
    pos_weight = None
    if use_class_weight:
        n_pos = float(np.sum(y_np == 1.0))
        n_neg = float(np.sum(y_np == -1.0))
        if n_pos > 0 and n_neg > 0:
            raw_w = n_neg / n_pos
            pos_weight = torch.tensor(raw_w ** cw_power, dtype=torch.float32)
            print(f"  [Baseline] pos_weight={pos_weight.item():.4f} "
                  f"(raw={raw_w:.4f}, power={cw_power}, "
                  f"n_pos={int(n_pos)}, n_neg={int(n_neg)})", flush=True)

    steps_per_pass = int(np.ceil(n_samples / batch_size))
    total_steps = total_eps * steps_per_pass
    warmup_fraction = config.get('WARMUP_FRACTION', 0.1) if config else 0.1
    scheduler = create_lr_scheduler(model.optimizer, total_steps, warmup_fraction)
    print(f"  [Baseline] LR schedule: warmup {int(total_steps*warmup_fraction)} steps "
          f"→ cosine decay over {total_steps} total steps", flush=True)

    patience = config.get('EARLY_STOPPING_PATIENCE', 15) if config else 15
    best_acc = 0.0
    no_improve_count = 0

    x_l = torch.tensor(X_tr, dtype=torch.float32)
    x_u = torch.tensor(X_tr, dtype=torch.float32)

    mon_file   = None
    mon_writer = None
    if log_path:
        mon_file = open(log_path, "w", newline="")
        mon_writer = csv.DictWriter(mon_file, fieldnames=[
            "pass", "avg_train_loss", "test_acc",
            "pred_pos", "pred_neg", "tp", "tn", "fp", "fn", "elapsed_s"
        ])
        mon_writer.writeheader()

    for pass_idx in range(total_eps):
        np.random.shuffle(indices)
        pass_loss   = 0.0
        pass_batches = 0
        for start in range(0, n_samples, batch_size):
            batch = indices[start:start + batch_size]
            loss_val = model.train_step(
                x_l[batch], x_u[batch],
                torch.tensor(y_np[batch], dtype=torch.float32),
                is_robust=False,
                pos_weight=pos_weight,
            )
            scheduler.step()
            pass_loss   += loss_val
            pass_batches += 1

        avg_loss = pass_loss / pass_batches if pass_batches > 0 else 0.0

        if (pass_idx + 1) % log_every == 0 or pass_idx == total_eps - 1:
            elapsed = time.time() - start_time
            dist = model.predict_distribution(X_te, y_test)
            row = {
                "pass": pass_idx + 1, "avg_train_loss": round(avg_loss, 6),
                "test_acc": round(dist["acc"], 4),
                "pred_pos": dist["pred_pos"], "pred_neg": dist["pred_neg"],
                "tp": dist["tp"], "tn": dist["tn"],
                "fp": dist["fp"], "fn": dist["fn"],
                "elapsed_s": round(elapsed, 1),
            }
            if mon_writer:
                mon_writer.writerow(row)
                mon_file.flush()
            print(
                f"  [Baseline pass {pass_idx+1:3d}/{total_eps}] "
                f"loss={avg_loss:.4f}  acc={dist['acc']:.4f}  "
                f"pred(+1={dist['pred_pos']}, -1={dist['pred_neg']})  "
                f"tp={dist['tp']} tn={dist['tn']} fp={dist['fp']} fn={dist['fn']}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )

            if dist["acc"] > best_acc + 1e-6:
                best_acc = dist["acc"]
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                print(f"  [Baseline] Early stopping at pass {pass_idx+1} — "
                      f"no improvement for {patience} checks", flush=True)
                break

    if mon_file:
        mon_file.close()

    final_dist = model.predict_distribution(X_te, y_test)
    best_acc = max(best_acc, final_dist["acc"])
    return best_acc, time.time() - start_time

def impute_examples_classification(X_current_state, example_indices_to_impute,
                                   imputation_method, original_dataset_context=None,
                                   col_stats=None, fitted_imputer=None,
                                   random_seed=None, X_train_gt=None):
    X_new_state = X_current_state.copy()
    if example_indices_to_impute is None or len(example_indices_to_impute) == 0:
        return X_new_state

    indices_arr = np.array(list(example_indices_to_impute))
    if indices_arr.size == 0:
        return X_new_state
    valid_indices = indices_arr[indices_arr < X_new_state.shape[0]]
    if imputation_method == 'gt' and X_train_gt is not None:
        valid_indices = valid_indices[valid_indices < X_train_gt.shape[0]]
    if valid_indices.size == 0:
        return X_new_state

    subset = X_new_state[valid_indices]
    imputed = subset.copy()

    if imputation_method == 'gt':
        if X_train_gt is not None:
            imputed = X_train_gt[valid_indices]
        else:
            print("Warning: GT requested but unavailable. Falling back to mean.")
            imputation_method = 'mean'

    if imputation_method == 'mean':
        if original_dataset_context is None or original_dataset_context.shape[0] == 0:
            imputed = np.nan_to_num(imputed, nan=0.5)
        else:
            imp = SimpleImputer(strategy='mean')
            complete_rows = ~np.isnan(original_dataset_context).any(axis=1)
            if np.any(complete_rows):
                imp.fit(original_dataset_context[complete_rows])
            else:
                imp.fit(np.nan_to_num(original_dataset_context))
            imputed = imp.transform(imputed)

    elif imputation_method == 'knn':
        if fitted_imputer is not None:
            imputed, _ = knn_imputation(imputed, "Transform", fitted_imputer)
        else:
            imputed = np.nan_to_num(imputed, nan=0.5)

    elif imputation_method == 'mice':
        if fitted_imputer is not None:
            imputed, _ = mice_impute(imputed, condition="Transform", imputer=fitted_imputer)
        else:
            imputed = np.nan_to_num(imputed, nan=0.5)

    elif imputation_method in ('rf_iterative', 'rf'):
        if fitted_imputer is not None:
            imputed, _ = rf_iterative_impute(imputed, condition="Transform", imputer=fitted_imputer)
        else:
            imputed = np.nan_to_num(imputed, nan=0.5)

    elif imputation_method != 'gt':
        imputed = np.nan_to_num(imputed, nan=0.5)

    X_new_state[valid_indices] = imputed
    return X_new_state

def run_single_experiment(original_dataset, labels, seed=None,
                          X_test_eval=None, y_test_eval=None,
                          X_original_complete_for_gt_imputation=None,
                          convergence_log_path=None,
                          imputation_method="mean",
                          mask_train=None,
                          bounds_train=None,
                          config=None,
                          num_incomplete_train=None):
    total_amr_time = 0.0
    try:
        max_total_epochs          = config.get('MAX_TOTAL_EPOCHS', 50)
        batch_size                = config.get('BATCH_SIZE', 64)
        train_epochs_per_iter     = config.get('TRAIN_EPOCHS_PER_ITER', 10)
        select_k_samples          = config.get('SELECT_K_SAMPLES', 10)
        loss_convergence_threshold = config.get('LOSS_CONVERGENCE_THRESHOLD', 0.01)
        hsas_threshold_divisor     = config.get('HSAS_REPAIR_THRESHOLD_DIVISOR', 1000.0)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        imputation_method = (imputation_method or "mean").lower()
        X_current  = np.array(original_dataset, dtype=np.float32)
        y_labels   = np.where(np.array(labels, dtype=np.float32).ravel() > 0, 1, -1)
        n_samples, n_features = X_current.shape

        robust_model = RobustFTTransformer(n_features=n_features, config=config)
        imputed_indices     = set()
        total_imputed_count = 0
        current_mask        = mask_train.copy() if mask_train is not None else np.isnan(X_current)
        current_bounds      = bounds_train.copy()

        if num_incomplete_train is None:
            num_incomplete_train = int(np.sum(np.any(current_mask, axis=1)))

        fitted_imputer = None
        knn_neighbors  = config.get('KNN_NEIGHBORS', 5)
        mice_max_iter  = config.get('MICE_MAX_ITER', 2)
        if imputation_method == 'knn':
            fitted_imputer, _ = knn_imputation(X_current, "Fit", None, knn_neighbors)
        elif imputation_method == 'mice':
            fitted_imputer, _ = mice_impute(X_current, mice_max_iter, seed, "Fit", None)
        elif imputation_method in ('rf_iterative', 'rf'):
            fitted_imputer, _ = rf_iterative_impute(np.nan_to_num(X_current), condition="Fit")

        use_class_weight = config.get('USE_CLASS_WEIGHT', True)
        cw_power = config.get('CLASS_WEIGHT_POWER', 0.5)
        pos_weight = None
        if use_class_weight:
            n_pos = float(np.sum(y_labels == 1))
            n_neg = float(np.sum(y_labels == -1))
            if n_pos > 0 and n_neg > 0:
                raw_w = n_neg / n_pos
                pos_weight = torch.tensor(raw_w ** cw_power, dtype=torch.float32)
                print(f"  [AMR] pos_weight={pos_weight.item():.4f} "
                      f"(raw={raw_w:.4f}, power={cw_power}, "
                      f"n_pos={int(n_pos)}, n_neg={int(n_neg)})", flush=True)

        steps_per_pass = int(np.ceil(n_samples / batch_size))
        total_sched_steps = max_total_epochs * train_epochs_per_iter * steps_per_pass
        warmup_fraction = config.get('WARMUP_FRACTION', 0.1)
        scheduler = create_lr_scheduler(robust_model.optimizer, total_sched_steps, warmup_fraction)
        print(f"  [AMR] LR schedule: warmup {int(total_sched_steps*warmup_fraction)} steps "
              f"→ cosine decay over {total_sched_steps} total steps", flush=True)

        patience = config.get('EARLY_STOPPING_PATIENCE', 15)
        best_test_acc = 0.0
        no_improve_count = 0

        conv_file   = None
        conv_writer = None
        if convergence_log_path:
            conv_file = open(convergence_log_path, "w", newline="")
            conv_writer = csv.DictWriter(conv_file, fieldnames=[
                "epoch", "nominal_loss",
                "hsas_avg", "hsas_min", "hsas_max", "hsas_std",
                "test_accuracy", "pred_pos", "pred_neg",
                "tp", "tn", "fp", "fn",
                "imputed_this_epoch", "total_imputed", "remaining_incomplete",
                "elapsed_s",
            ])
            conv_writer.writeheader()

        test_acc   = 0.0
        hsas_val   = 0.0
        final_epoch = 0
        indices     = np.arange(n_samples)

        for epoch in range(max_total_epochs):
            t0 = time.time()
            hsas_scores         = np.zeros(n_samples)
            candidate_mask      = current_mask.any(axis=1)
            candidate_indices   = np.where(candidate_mask)[0]

            if len(candidate_indices) > 0:
                hsas_vals = robust_model.compute_hsas(
                    current_bounds[candidate_indices, :, 0],
                    current_bounds[candidate_indices, :, 1],
                    y_labels[candidate_indices],
                    batch_size=batch_size,
                )
                hsas_scores[candidate_indices] = hsas_vals
            hsas_val = np.mean(hsas_scores)
            total_amr_time += time.time() - t0

            t0 = time.time()
            imputed_this_epoch = 0
            not_yet_imputed = np.where(
                (~np.isin(np.arange(n_samples), list(imputed_indices))) & candidate_mask
            )[0]

            if len(not_yet_imputed) > 0:
                cand_hsas = hsas_scores[not_yet_imputed]
                repair_threshold = loss_convergence_threshold / hsas_threshold_divisor
                valid_mask = cand_hsas > repair_threshold
                if np.any(valid_mask):
                    valid_cands = not_yet_imputed[valid_mask]
                    valid_hsas  = cand_hsas[valid_mask]
                    top_k_idx   = np.argsort(-valid_hsas)[:select_k_samples]
                    to_repair   = valid_cands[top_k_idx]

                    X_current = impute_examples_classification(
                        X_current, to_repair, imputation_method,
                        original_dataset_context=original_dataset,
                        fitted_imputer=fitted_imputer,
                        random_seed=seed,
                        X_train_gt=X_original_complete_for_gt_imputation,
                    )
                    for r_idx in to_repair:
                        imputed_indices.add(r_idx)
                        total_imputed_count += 1
                        imputed_this_epoch  += 1
                        repaired_val = X_current[r_idx]
                        current_bounds[r_idx, :, 0] = repaired_val
                        current_bounds[r_idx, :, 1] = repaired_val
                        current_mask[r_idx, :] = False
            total_amr_time += time.time() - t0

            t0 = time.time()
            epoch_loss   = 0.0
            num_batches  = 0
            updated_candidate_mask = current_mask.any(axis=1)

            for _ in range(train_epochs_per_iter):
                np.random.shuffle(indices)
                for start in range(0, n_samples, batch_size):
                    batch_idx = indices[start:start + batch_size]
                    x_l_b = torch.tensor(current_bounds[batch_idx, :, 0], dtype=torch.float32)
                    x_u_b = torch.tensor(current_bounds[batch_idx, :, 1], dtype=torch.float32)
                    y_b   = torch.tensor(y_labels[batch_idx], dtype=torch.float32)

                    batch_needs_robust = np.any(updated_candidate_mask[batch_idx])
                    loss_val = robust_model.train_step(x_l_b, x_u_b, y_b,
                                                       is_robust=batch_needs_robust,
                                                       pos_weight=pos_weight)
                    scheduler.step()
                    epoch_loss  += loss_val
                    num_batches += 1
            total_amr_time += time.time() - t0

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            cand_hsas_vals = hsas_scores[hsas_scores > 0]
            hsas_min = float(np.min(cand_hsas_vals)) if len(cand_hsas_vals) > 0 else 0.0
            hsas_max = float(np.max(cand_hsas_vals)) if len(cand_hsas_vals) > 0 else 0.0
            hsas_std = float(np.std(cand_hsas_vals)) if len(cand_hsas_vals) > 0 else 0.0

            dist     = robust_model.predict_distribution(X_test_eval, y_test_eval)
            test_acc = dist["acc"]
            remaining = int(np.sum(current_mask.any(axis=1)))

            if conv_writer:
                conv_writer.writerow({
                    "epoch": epoch + 1, "nominal_loss": round(avg_loss, 6),
                    "hsas_avg": round(hsas_val, 6),
                    "hsas_min": round(hsas_min, 6), "hsas_max": round(hsas_max, 6),
                    "hsas_std": round(hsas_std, 6),
                    "test_accuracy": round(test_acc, 4),
                    "pred_pos": dist["pred_pos"], "pred_neg": dist["pred_neg"],
                    "tp": dist["tp"], "tn": dist["tn"],
                    "fp": dist["fp"], "fn": dist["fn"],
                    "imputed_this_epoch": imputed_this_epoch,
                    "total_imputed": total_imputed_count,
                    "remaining_incomplete": remaining,
                    "elapsed_s": round(total_amr_time, 1),
                })
                conv_file.flush()

            print(
                f"  [AMR epoch {epoch+1}/{max_total_epochs}] "
                f"loss={avg_loss:.4f}  hsas_avg={hsas_val:.4f} "
                f"(min={hsas_min:.4f} max={hsas_max:.4f})  "
                f"acc={test_acc:.4f}  pred(+1={dist['pred_pos']}, -1={dist['pred_neg']})  "
                f"repaired={imputed_this_epoch}  remaining={remaining}",
                flush=True,
            )

            if test_acc > best_test_acc + 1e-6:
                best_test_acc = test_acc
                no_improve_count = 0
            else:
                no_improve_count += 1

            final_epoch = epoch + 1
            if epoch >= 0 and hsas_val < loss_convergence_threshold:
                print(f"  [AMR] Converged at epoch {final_epoch} — HSAS {hsas_val:.6f} < threshold {loss_convergence_threshold}")
                break
            if no_improve_count >= patience:
                print(f"  [AMR] Early stopping at epoch {final_epoch} — "
                      f"no accuracy improvement for {patience} epochs", flush=True)
                break

        if conv_file:
            conv_file.close()

        best_test_acc = max(best_test_acc, test_acc)
        ratio = total_imputed_count / num_incomplete_train if num_incomplete_train > 0 else 0.0
        return {
            "total_time": total_amr_time,
            "total_imputed_count": total_imputed_count,
            "imputation_ratio": ratio,
            "test_accuracy": best_test_acc,
            "uncertainty_range": hsas_val,
            "total_epochs": final_epoch,
            "imputed_indices": list(imputed_indices),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "total_time": 0.0, "total_imputed_count": 0,
            "imputation_ratio": 0.0, "test_accuracy": 0.0,
            "uncertainty_range": 0.0, "total_epochs": 0,
            "imputed_indices": [],
        }

def run_acm_on_csv_file_with_imputation_method(
    csv_file_path, imputation_method, output_dir=None,
    seeds_to_try=None, config=None, test_csv_path=None
):
    if config is None:
        config = load_config('config_ft_transformer.json')

    (X_train_inc, y_train, X_test, y_test, dataset_name, mask_train,
     bounds_train, num_incomplete_train, X_train_gt) = load_dataset_from_csv(
        csv_file_path, config=config, test_csv_path=test_csv_path
    )

    base_dataset_name = dataset_name.split('_')[0]
    if base_dataset_name == 'default':
        base_dataset_name = 'credit_default'
    config = load_config('config_ft_transformer.json', dataset_name=base_dataset_name)

    run_baseline  = config.get('global_settings', {}).get('run_impute_all_baseline', True)
    baseline_acc  = 0.0
    baseline_time = 0.0

    if run_baseline:
        baseline_acc, baseline_time = run_ft_baseline_impute_all(
            X_train_inc, y_train, X_test, y_test,
            method=imputation_method, config=config, X_train_gt=X_train_gt,
            seed=config.get('RANDOM_STATE', 42)
        )
        print(f"\nBaseline — FT-Transformer impute-all ({imputation_method}): "
              f"acc={baseline_acc:.4f}, time={baseline_time:.2f}s")

    if output_dir is None:
        output_dir = config.get('global_settings', {}).get(
            'output_dir', './ACM_Results_FTTransformer'
        )

    iter_log_dir   = os.path.join(output_dir, "Iter_Logs")
    results_dir    = os.path.join(output_dir, "Final_Results")
    os.makedirs(iter_log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    if seeds_to_try is None:
        num_seeds    = config.get('NUM_RANDOM_SEEDS', 3)
        seeds_to_try = generate_random_seeds(num_seeds)

    results = []
    epoch_log_path = os.path.join(results_dir, f"{imputation_method}_{dataset_name}_epoch_log.csv")

    with open(epoch_log_path, "w", newline="") as ef:
        writer = csv.DictWriter(ef, fieldnames=[
            "run", "total_time", "total_imputed_count", "imputation_ratio",
            "test_accuracy", "baseline_acc", "baseline_time",
            "uncertainty_range", "total_epochs"
        ])
        writer.writeheader()

        for i, run_seed in enumerate(seeds_to_try):
            print(f"\nMethod: {imputation_method}, Run {i+1}/{len(seeds_to_try)} "
                  f"with seed {run_seed}")
            conv_log = os.path.join(iter_log_dir,
                                    f"{imputation_method}_{dataset_name}_run{i+1}_convergence.csv")
            result = run_single_experiment(
                X_train_inc, y_train, seed=run_seed,
                X_test_eval=X_test, y_test_eval=y_test,
                X_original_complete_for_gt_imputation=X_train_gt,
                convergence_log_path=conv_log,
                imputation_method=imputation_method,
                mask_train=mask_train, bounds_train=bounds_train,
                config=config, num_incomplete_train=num_incomplete_train,
            )
            results.append(result)
            row = {"run": i + 1, "baseline_acc": baseline_acc, "baseline_time": baseline_time}
            for k, v in result.items():
                if k != "imputed_indices":
                    row[k] = v
            writer.writerow(row)

        metrics = ["total_time", "total_imputed_count", "imputation_ratio",
                   "test_accuracy", "uncertainty_range", "total_epochs"]
        avg_row = {"run": "average", "baseline_acc": baseline_acc, "baseline_time": baseline_time}
        for m in metrics:
            vals = [r[m] for r in results]
            avg_row[m] = f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}"
        writer.writerow(avg_row)

    best_acc = max(r["test_accuracy"] for r in results)
    print(f"\n{'='*55}")
    print(f"  FT-Transformer AMR  best accuracy : {best_acc:.4f}")
    print(f"  FT-Transformer impute-all baseline: {baseline_acc:.4f}  "
          f"(delta AMR-baseline: {best_acc - baseline_acc:+.4f})")
    print(f"  Imputation ratio (last run)       : {results[-1]['imputation_ratio']:.4f} "
          f"({results[-1]['total_imputed_count']}/{num_incomplete_train})")
    print(f"{'='*55}")

    return {
        "dataset_name": dataset_name,
        "imputation_method": imputation_method,
        "results": results,
        "best_accuracy": best_acc,
        "baseline_accuracy": baseline_acc,
    }

def run_acm_on_csv_file(csv_file_path, output_dir=None, seeds_to_try=None,
                        imputation_methods=None, config=None, test_csv_path=None):
    if config is None:
        config = load_config('config_ft_transformer.json')
    if output_dir is None:
        output_dir = config.get('global_settings', {}).get(
            'output_dir', './ACM_Results_FTTransformer'
        )
    if seeds_to_try is None:
        num_seeds    = config.get('NUM_RANDOM_SEEDS', 3)
        seeds_to_try = generate_random_seeds(num_seeds)
    if imputation_methods is None:
        imputation_methods = config.get('global_settings', {}).get(
            'imputation_methods', config.get('IMPUTATION_METHODS', ['knn'])
        )

    dataset_name = os.path.basename(csv_file_path).replace(".csv", "")
    all_results  = {}
    for imp_method in imputation_methods:
        try:
            r = run_acm_on_csv_file_with_imputation_method(
                csv_file_path, imp_method, output_dir,
                seeds_to_try, config=config, test_csv_path=test_csv_path
            )
            all_results[imp_method] = r
        except Exception as e:
            print(f"Error in {dataset_name} with {imp_method}: {e}")
    return {"dataset_name": dataset_name, "imputation_results": all_results}

DATASET_CONFIGS = {
    "malware": {
        "default_file": "Data_LL/malware_MNAR_train_60.csv",
        "test_file":    "Data_LL/malware_test.csv",
        "variants_dir": "Data_LL", "prefix": "malware",
    },
    "tuadromd": {
        "default_file": "Data_LL/tuadromd_MNAR_train_60.csv",
        "test_file":    "Data_LL/tuadromd_test.csv",
        "variants_dir": "Data_LL", "prefix": "tuadromd",
    },
    "credit_default": {
        "default_file": "Data_LL/default_MNAR_train_60.csv",
        "test_file":    "Data_LL/default_test.csv",
        "variants_dir": "Data_LL", "prefix": "default",
    },
    "fraud": {
        "default_file": "Data_LL/fraud_MCAR_train_20.csv",
        "test_file":    "Data_LL/fraud_test.csv",
        "variants_dir": "Data_LL", "prefix": "fraud",
    },
    "higgs": {
        "default_file": os.path.join(os.environ.get("HIGGS_DATA_DIR", "/path/to/higgs/data"), "higgs_MCAR_train_20.csv"),
        "test_file":    os.path.join(os.environ.get("HIGGS_DATA_DIR", "/path/to/higgs/data"), "higgs_test.csv"),
        "variants_dir": os.environ.get("HIGGS_DATA_DIR", "/path/to/higgs/data"),
        "prefix": "higgs",
    },
    "breast": {"default_file": "Minimal-Imputation/Synthetic-Datasets/breast.csv"},
    "water":  {"default_file": "Minimal-Imputation/Synthetic-Datasets/water.csv"},
    "online": {"default_file": "Minimal-Imputation/Synthetic-Datasets/online.csv"},
    "bankruptcy": {"default_file": "Minimal-Imputation/Synthetic-Datasets/bankrupt_normalized.csv"},
}

def parse_args():
    parser = argparse.ArgumentParser(description="FT-Transformer AMR (SW-RGS)")
    parser.add_argument("--dataset", type=str, nargs="+", help="Dataset name(s)")
    parser.add_argument("--single",  type=str, help="Path to a single CSV file")
    parser.add_argument("--output",  type=str, help="Output directory override")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config('config_ft_transformer.json')
    global_settings = config.get('global_settings', {})
    output_dir = args.output or global_settings.get('output_dir', './ACM_Results_FTTransformer')

    datasets_to_run = []
    if args.single:
        datasets_to_run = [args.single]
    elif args.dataset:
        for ds in args.dataset:
            found_variant = False
            for base in ["malware", "tuadromd", "credit_default", "default", "fraud", "higgs"]:
                if ds.startswith(base + "_"):
                    datasets_to_run.append(ds)
                    found_variant = True
                    break
            if not found_variant:
                if ds in DATASET_CONFIGS:
                    datasets_to_run.append(ds)
                else:
                    print(f"Warning: Dataset {ds} not found in DATASET_CONFIGS")
    else:
        datasets_to_run = global_settings.get('datasets_to_run', list(DATASET_CONFIGS.keys()))

    for ds_name in datasets_to_run:
        csv_file = None
        test_csv_path = None
        is_variant = False

        for base_ds, info in DATASET_CONFIGS.items():
            if "variants_dir" in info and (
                ds_name.startswith(base_ds + "_")
                or (base_ds == 'credit_default' and ds_name.startswith("default_"))
            ):
                parts = ds_name.split('_')
                if len(parts) >= 3:
                    mechanism = parts[-2]
                    factor    = parts[-1]
                    prefix    = info["prefix"]
                    csv_file  = os.path.join(info["variants_dir"],
                                             f"{prefix}_{mechanism}_train_{factor}.csv")
                    test_csv_path = info["test_file"]
                    is_variant = True
                    break

        if not is_variant:
            if os.path.exists(ds_name) and ds_name.endswith('.csv'):
                csv_file = ds_name
            elif ds_name in DATASET_CONFIGS:
                csv_file      = DATASET_CONFIGS[ds_name]["default_file"]
                test_csv_path = DATASET_CONFIGS[ds_name].get("test_file")
            else:
                continue

        if not csv_file or not os.path.exists(csv_file):
            print(f"Warning: CSV file not found for {ds_name}: {csv_file}")
            continue

        print(f"\n{'='*60}\nProcessing dataset: {csv_file}\n{'='*60}")
        base_ds_name = ds_name.split('_')[0]
        if base_ds_name == 'default':
            base_ds_name = 'credit_default'
        ds_config = load_config(
            'config_ft_transformer.json',
            dataset_name=base_ds_name if base_ds_name in DATASET_CONFIGS else None
        )
        run_acm_on_csv_file(csv_file, output_dir, config=ds_config,
                            test_csv_path=test_csv_path)
