"""
SASE — State-Aware Stacking Ensemble
=====================================
Main modeling script for all four Amazon states (Pará, Mato Grosso, Amazonas,
Roraima). This script reproduces the headline results of the paper.

Split strategy (spatially-aware via GroupShuffleSplit over spatial blocks):
  70% train   → fit base learners + meta-learner
  15% val     → optimize decision threshold τ*
  15% test    → report final F1 and AUC-ROC (never touched before)

Runs N_SPLITS independent splits and reports mean ± std.

Reproducibility
---------------
With the default parameters (BLOCK_RES=0.2, RANDOM_SEED=42, N_SPLITS=5) and the
provided dados_concatenados.csv, the aggregate test-set F1 scores are:
    Pará 0.881 | Mato Grosso 0.742 | Amazonas 0.520 | Roraima 0.478
matching results/reference_res020/sase_summary.csv.

Configuration is read from config.py (DATA_CSV, RESULTS_DIR) with CLI overrides.

Outputs per state (saved to OUTPUT_DIR):
  - {STATE}_classification_report.txt
  - {STATE}_roc_curve.png / {STATE}_pr_curve.png
  - {STATE}_threshold_curve.png / {STATE}_confusion_matrix.png
  - {STATE}_all_splits.csv
Global summary:
  - sase_summary.csv
Cross-state generalization (Pará → others):
  - generalization_summary.csv / generalization_report.txt

Usage
-----
    python src/sase_spatial.py
    python src/sase_spatial.py --block-res 0.2 --n-splits 5 --seed 42
    DATA_DIR=/path/to/data RESULTS_DIR=/path/to/out python src/sase_spatial.py
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import json
from shapely.geometry import shape
from sklearn.model_selection import (
    GroupShuffleSplit, cross_val_predict, StratifiedKFold, StratifiedGroupKFold
)
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Import repo-level configuration (portable paths, no hard-coded machine paths).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_CSV, RESULTS_DIR

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION  (paper defaults; override via CLI / env vars)
# ============================================================

TARGET          = "VPRESSAO"
GROUP_COL       = "CODEALERTA"
GEO_COL         = ".geo"      # GeoJSON column (Polygon/MultiPolygon, WGS84)
BLOCK_COL       = "_block"    # spatial block id derived from polygon centroid
BLOCK_RES       = 0.2         # grid resolution in DEGREES (~22 km). Paper value: 0.2
RANDOM_SEED     = 42
N_SPLITS        = 5
N_META_FOLDS    = 5           # OOF folds to train the meta-learner (leak-free stacking)
THRESHOLD_RANGE = np.linspace(0.2, 0.8, 61)

# Split proportions: 70% train | 15% val | 15% test
VAL_TEST_SIZE   = 0.30    # step 1: 70% train, 30% val+test
TEST_FROM_VT    = 0.50    # step 2: 50% of val+test → test (15%), rest → val (15%)

LABEL_MAP = {"ilegal_mining": 1, "agriculture": 0, "resto": 0}

STATE_FEATURES = {
    "PARÁ": [
        "diss_vv_mean", "contrast_vv_mean", "contrast_vv_p10", "diss_vv_p10",
        "diss_vv_p90", "contrast_vv_p90",
        "corr_vh_mean", "corr_vh_p10", "diss_vh_mean", "diss_vh_p10",
        "corr_vh_p90", "contrast_vh_p10", "contrast_vh_mean",
        "diss_vh_p90", "contrast_vh_p90",
        "VV_stdDev", "VH_stdDev",
        "corr_vv_p10", "corr_vv_mean",
        "tpi",
        "VV_minus_VH_mean", "VV_minus_VH_p10", "VV_minus_VH_p90"
    ],
    "MATO GROSSO": [
        "contrast_vv_mean", "diss_vv_mean", "diss_vv_p90", "contrast_vv_p10",
        "contrast_vv_p90", "diss_vv_p10",
        "corr_vh_mean", "VV_stdDev", "diss_vh_mean", "VH_stdDev",
        "corr_vh_p10", "contrast_vh_mean", "contrast_vh_p10", "diss_vh_p10",
        "VV_minus_VH_p90", "corr_vh_p90", "diss_vh_p90",
        "VV_minus_VH_mean", "contrast_vh_p90",
        "corr_vv_mean", "corr_vv_p10", "VV_minus_VH_p10"
    ],
    "AMAZONAS": [
        "contrast_vv_mean", "diss_vv_mean", "diss_vv_p90",
        "contrast_vv_p90", "contrast_vv_p10", "diss_vv_p10",
        "corr_vh_p90", "diss_vh_p90", "diss_vh_mean", "contrast_vh_mean",
        "VV_stdDev", "VH_stdDev",
        "corr_vv_p90", "corr_vh_mean", "corr_vv_mean",
        "dist_river", "slope", "roughness"
    ],
    "RORAIMA": [
        "contrast_vv_mean", "diss_vv_mean", "contrast_vv_p90",
        "diss_vv_p90", "contrast_vv_p10", "diss_vv_p10",
        "VV_stdDev", "VH_stdDev",
        "corr_vv_mean", "corr_vv_p10", "corr_vv_p90",
        "corr_vh_mean", "corr_vh_p10", "corr_vh_p90",
        "dist_river"
    ]
}

STATE_USE_XGB = {"PARÁ": True, "MATO GROSSO": True, "AMAZONAS": True, "RORAIMA": True}
STATE_USE_MLP = {"PARÁ": True, "MATO GROSSO": True, "AMAZONAS": True, "RORAIMA": True}

# Filled in by main() from CLI/config before run.
DATA_PATH  = str(DATA_CSV)
OUTPUT_DIR = str(RESULTS_DIR / "sase_res020")


# ============================================================
# HELPERS
# ============================================================

def log_transform(X, cols):
    X = X.copy()
    for col in cols:
        if col in X.columns:
            X[col] = np.log1p(X[col])
    return X

def get_log_cols(cols):
    return [c for c in cols if "contrast" in c or "diss" in c]

def build_base_models(y_train, use_xgb=True, use_mlp=True):
    pos     = (y_train == 1).sum()
    neg     = (y_train == 0).sum()
    scale_w = neg / pos if pos > 0 else 1.0

    models = {}
    models["RF"] = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=RANDOM_SEED
    )
    if use_xgb:
        models["XGB"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_w,
            eval_metric="logloss", random_state=RANDOM_SEED
        )
    models["LR"] = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED
        ))
    ])
    if use_mlp:
        models["MLP"] = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=500, random_state=RANDOM_SEED
            ))
        ])
    return models

def get_meta_features(models, X):
    return np.column_stack([
        m.predict_proba(X)[:, 1] for m in models.values()
    ])

def compute_blocks(df, geo_col=GEO_COL, res=BLOCK_RES):
    """
    Derive a spatial block id per row from the polygon centroid (GeoJSON column).
    Block = cell of a regular grid of `res` degrees. Polygons in the same block
    always fall in the same partition (train/val/test), enforcing a geographic
    separation between partitions (Roberts et al., 2017).
    """
    ids = []
    for raw in df[geo_col]:
        try:
            g = raw if isinstance(raw, dict) else json.loads(raw)
            c = shape(g).centroid
            bx = int(np.floor(c.x / res))
            by = int(np.floor(c.y / res))
            ids.append(f"{bx}_{by}")
        except Exception:
            ids.append(np.nan)
    return pd.Series(ids, index=df.index)

def get_oof_meta_features(models, X, y, groups, n_folds=N_META_FOLDS, seed=RANDOM_SEED):
    """
    Generate OUT-OF-FOLD meta-features to train the meta-learner, with folds
    GROUPED BY SPATIAL BLOCK (StratifiedGroupKFold). Each sample gets the
    probability from a model that never saw any sample from its block, avoiding
    both in-sample leakage and spatial leakage at level 2 of the stacking.

    Safeguard: the number of folds is bounded by the number of blocks containing
    positives; if fewer than 2 positive-bearing blocks exist, it falls back to
    non-grouped StratifiedKFold (with a warning).
    """
    n_pos = int((y == 1).sum())
    pos_blocks = groups[y.values == 1].nunique()

    oof = []
    if pos_blocks >= 2:
        folds = max(2, min(n_folds, pos_blocks))
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        try:
            for model in models.values():
                proba = cross_val_predict(
                    clone(model), X, y, groups=groups, cv=cv,
                    method="predict_proba", n_jobs=-1
                )[:, 1]
                oof.append(proba)
            return np.column_stack(oof)
        except Exception as e:
            print(f"    [warn] StratifiedGroupKFold failed ({e}); "
                  f"falling back to non-grouped StratifiedKFold.")
            oof = []

    # fallback: not enough positive-bearing blocks
    print(f"    [warn] only {pos_blocks} block(s) with positives in train; "
          f"non-grouped OOF.")
    folds = max(2, min(n_folds, n_pos))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for model in models.values():
        proba = cross_val_predict(
            clone(model), X, y, cv=skf, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof.append(proba)
    return np.column_stack(oof)

def optimize_threshold(y_true, y_prob):
    """Find best threshold on VALIDATION set only."""
    best_f1, best_t = 0.0, 0.5
    for t in THRESHOLD_RANGE:
        pred = (y_prob > t).astype(int)
        f1   = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return best_t, best_f1

def plot_roc(y_true, y_prob, state, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{state} — ROC Curve (SASE, best split)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{state}_roc_curve.png"), dpi=150)
    plt.close()

def plot_pr(y_true, y_prob, state, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AUC-PR = {auc_pr:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{state} — Precision-Recall Curve (SASE, best split)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{state}_pr_curve.png"), dpi=150)
    plt.close()

def plot_threshold_curve(y_val, y_prob_val, best_t, state, output_dir):
    """Plot threshold curve on VALIDATION set (where τ* was chosen)."""
    f1_scores = [f1_score(y_val, (y_prob_val > t).astype(int), zero_division=0)
                 for t in THRESHOLD_RANGE]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(THRESHOLD_RANGE, f1_scores, lw=2)
    ax.axvline(best_t, color="red", linestyle="--",
               label=f"Best threshold = {best_t:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-Score (validation set)")
    ax.set_title(f"{state} — Threshold Optimization on Validation Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{state}_threshold_curve.png"), dpi=150)
    plt.close()

def plot_confusion(y_true, y_pred, state, output_dir):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["resto (0)", "ilegal_mining (1)"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{state} — Confusion Matrix (test set, best split)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{state}_confusion_matrix.png"), dpi=150)
    plt.close()


# ============================================================
# SINGLE SPLIT RUNNER
# ============================================================

def run_single_split(X, y, groups, seed, use_xgb, use_mlp):
    """
    Three-way spatial split:
      70% train | 15% validation | 15% test
    - τ* is optimized on the validation set
    - Final metrics reported on the test set only
    """

    # Step 1: 70% train, 30% val+test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_TEST_SIZE,
                             random_state=seed)
    train_idx, val_test_idx = next(gss1.split(X, y, groups))

    X_train  = X.iloc[train_idx]
    y_train  = y.iloc[train_idx]
    X_vt     = X.iloc[val_test_idx]
    y_vt     = y.iloc[val_test_idx]
    grp_vt   = groups.iloc[val_test_idx]

    # Step 2: split val+test into 50/50 → 15% val, 15% test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=TEST_FROM_VT,
                             random_state=seed + 100)
    val_idx, test_idx = next(gss2.split(X_vt, y_vt, grp_vt))

    X_val  = X_vt.iloc[val_idx]
    y_val  = y_vt.iloc[val_idx]
    X_test = X_vt.iloc[test_idx]
    y_test = y_vt.iloc[test_idx]

    # spatial blocks per partition
    grp_train = groups.iloc[train_idx]
    grp_val   = grp_vt.iloc[val_idx]
    grp_test  = grp_vt.iloc[test_idx]

    # Level 1 — base learners
    base_models = build_base_models(y_train, use_xgb=use_xgb, use_mlp=use_mlp)

    # OUT-OF-FOLD meta-features with folds grouped by spatial block.
    meta_X_train = get_oof_meta_features(base_models, X_train, y_train, grp_train,
                                         n_folds=N_META_FOLDS, seed=seed)

    # Only now refit base learners on the FULL training set, to infer on
    # val/test (these sets remain strictly held out).
    for model in base_models.values():
        model.fit(X_train, y_train)

    meta_X_val   = get_meta_features(base_models, X_val)
    meta_X_test  = get_meta_features(base_models, X_test)

    # Level 2 — meta-learner trained on OOF predictions
    meta_model = LogisticRegression(class_weight="balanced",
                                    random_state=RANDOM_SEED)
    meta_model.fit(meta_X_train, y_train)

    # Optimize threshold on VALIDATION set
    y_prob_val         = meta_model.predict_proba(meta_X_val)[:, 1]
    best_t, _          = optimize_threshold(y_val, y_prob_val)

    # Evaluate on TEST set using τ* from validation
    y_prob_test        = meta_model.predict_proba(meta_X_test)[:, 1]
    y_pred_test        = (y_prob_test > best_t).astype(int)

    f1     = f1_score(y_test, y_pred_test, zero_division=0)
    auc    = roc_auc_score(y_test, y_prob_test)
    auc_pr = average_precision_score(y_test, y_prob_test)

    prec_mining = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
    rec_mining  = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
    prec_resto  = precision_score(y_test, y_pred_test, pos_label=0, zero_division=0)
    rec_resto   = recall_score(y_test, y_pred_test, pos_label=0, zero_division=0)

    return {
        "f1":           f1,
        "auc":          auc,
        "auc_pr":       auc_pr,
        "threshold":    best_t,
        "prec_mining":  prec_mining,
        "rec_mining":   rec_mining,
        "prec_resto":   prec_resto,
        "rec_resto":    rec_resto,
        "y_val":        y_val,
        "y_prob_val":   y_prob_val,
        "y_test":       y_test,
        "y_prob_test":  y_prob_test,
        "y_pred_test":  y_pred_test,
        "base_models":  base_models,
        "meta_model":   meta_model,
        "n_train":      len(y_train),
        "n_val":        len(y_val),
        "n_test":       len(y_test),
        # mining/resto composition and number of blocks per spatial partition
        "train_pos":    int((y_train == 1).sum()),
        "train_neg":    int((y_train == 0).sum()),
        "val_pos":      int((y_val == 1).sum()),
        "val_neg":      int((y_val == 0).sum()),
        "test_pos":     int((y_test == 1).sum()),
        "test_neg":     int((y_test == 0).sum()),
        "train_minpct": float(100 * (y_train == 1).mean()),
        "val_minpct":   float(100 * (y_val == 1).mean()),
        "test_minpct":  float(100 * (y_test == 1).mean()),
        "n_blocks_train": int(grp_train.nunique()),
        "n_blocks_val":   int(grp_val.nunique()),
        "n_blocks_test":  int(grp_test.nunique()),
    }


# ============================================================
# MAIN PIPELINE — PER STATE
# ============================================================

def run_state(state, df, features, use_xgb, use_mlp):
    print(f"\n{'='*60}")
    print(f"  STATE: {state}  |  {N_SPLITS} splits  |  70/15/15")
    print(f"{'='*60}")

    df_s     = df[df["ESTADO"] == state].copy()
    features = [f for f in features if f in df_s.columns]

    df_model = df_s[features + [TARGET, GROUP_COL, BLOCK_COL]].dropna()
    X        = df_model[features].copy()
    y        = df_model[TARGET].map(LABEL_MAP)
    groups   = df_model[BLOCK_COL]   # SPATIAL blocking

    n_blocks = groups.nunique()
    print(f"  Total: {len(y)}  |  Pos: {(y==1).sum()}  |  Neg: {(y==0).sum()}  "
          f"|  Spatial blocks ({BLOCK_RES}°): {n_blocks}")

    log_cols = get_log_cols(features)
    X        = log_transform(X, log_cols)

    split_metrics = []
    best_split    = None
    best_f1_seen  = -1.0

    for split_i in range(N_SPLITS):
        seed_i = RANDOM_SEED + split_i
        result = run_single_split(X, y, groups, seed_i, use_xgb, use_mlp)
        result["split"] = split_i
        result["seed"]  = seed_i
        split_metrics.append(result)

        print(f"  Split {split_i+1}/{N_SPLITS} "
              f"(seed={seed_i}) "
              f"train={result['n_train']} "
              f"val={result['n_val']} "
              f"test={result['n_test']} | "
              f"τ*={result['threshold']:.2f}  "
              f"F1={result['f1']:.4f}  "
              f"AUC={result['auc']:.4f}")
        # mining vs resto composition per spatial fold
        print(f"      ├─ train: {result['train_pos']} min / {result['train_neg']} resto "
              f"({result['train_minpct']:.1f}% min, {result['n_blocks_train']} blocks)")
        print(f"      ├─ val  : {result['val_pos']} min / {result['val_neg']} resto "
              f"({result['val_minpct']:.1f}% min, {result['n_blocks_val']} blocks)")
        print(f"      └─ test : {result['test_pos']} min / {result['test_neg']} resto "
              f"({result['test_minpct']:.1f}% min, {result['n_blocks_test']} blocks)")

        if result["f1"] > best_f1_seen:
            best_f1_seen = result["f1"]
            best_split   = result

    # --------------------------------------------------------
    # Aggregate stats across splits
    # --------------------------------------------------------
    def _m(key): return float(np.mean([m[key] for m in split_metrics]))
    def _s(key): return float(np.std([m[key]  for m in split_metrics]))

    summary = {
        "state":            state,
        "n_total":          len(y),
        "n_positive":       int((y == 1).sum()),
        "n_negative":       int((y == 0).sum()),
        "n_splits":         N_SPLITS,
        "split_strategy":   "70/15/15 (train/val/test)",
        "f1_mean":          round(_m("f1"),           4),
        "f1_std":           round(_s("f1"),            4),
        "auc_mean":         round(_m("auc"),           4),
        "auc_std":          round(_s("auc"),            4),
        "auc_pr_mean":      round(_m("auc_pr"),         4),
        "auc_pr_std":       round(_s("auc_pr"),         4),
        "prec_mining_mean": round(_m("prec_mining"),   4),
        "prec_mining_std":  round(_s("prec_mining"),    4),
        "rec_mining_mean":  round(_m("rec_mining"),    4),
        "rec_mining_std":   round(_s("rec_mining"),     4),
        "prec_resto_mean":  round(_m("prec_resto"),    4),
        "prec_resto_std":   round(_s("prec_resto"),     4),
        "rec_resto_mean":   round(_m("rec_resto"),     4),
        "rec_resto_std":    round(_s("rec_resto"),      4),
        "threshold_mean":   round(_m("threshold"),     2),
        "threshold_std":    round(_s("threshold"),      2),
    }

    print(f"\n  ── {N_SPLITS}-split aggregate (test set) ──")
    print(f"  F1     : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  AUC    : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"  AUC-PR : {summary['auc_pr_mean']:.4f} ± {summary['auc_pr_std']:.4f}")
    print(f"  Prec(mining): {summary['prec_mining_mean']:.4f} ± {summary['prec_mining_std']:.4f}")
    print(f"  Rec (mining): {summary['rec_mining_mean']:.4f} ± {summary['rec_mining_std']:.4f}")

    # --------------------------------------------------------
    # Save per-split CSV
    # --------------------------------------------------------
    state_safe = state.replace(" ", "_")
    pd.DataFrame([{
        "split":       m["split"],
        "seed":        m["seed"],
        "n_train":     m["n_train"],
        "n_val":       m["n_val"],
        "n_test":      m["n_test"],
        "train_pos":   m["train_pos"],
        "train_neg":   m["train_neg"],
        "train_minpct": round(m["train_minpct"], 2),
        "val_pos":     m["val_pos"],
        "val_neg":     m["val_neg"],
        "val_minpct":  round(m["val_minpct"], 2),
        "test_pos":    m["test_pos"],
        "test_neg":    m["test_neg"],
        "test_minpct": round(m["test_minpct"], 2),
        "n_blocks_train": m["n_blocks_train"],
        "n_blocks_val":   m["n_blocks_val"],
        "n_blocks_test":  m["n_blocks_test"],
        "threshold":   m["threshold"],
        "f1":          m["f1"],
        "auc":         m["auc"],
        "auc_pr":      m["auc_pr"],
        "prec_mining": m["prec_mining"],
        "rec_mining":  m["rec_mining"],
        "prec_resto":  m["prec_resto"],
        "rec_resto":   m["rec_resto"],
    } for m in split_metrics]).to_csv(
        os.path.join(OUTPUT_DIR, f"{state_safe}_all_splits.csv"), index=False
    )

    # --------------------------------------------------------
    # Report and plots for best split
    # --------------------------------------------------------
    report = classification_report(
        best_split["y_test"], best_split["y_pred_test"],
        target_names=["resto (0)", "ilegal_mining (1)"],
        digits=4
    )
    with open(os.path.join(OUTPUT_DIR,
              f"{state_safe}_classification_report.txt"), "w") as f:
        f.write(f"STATE: {state}\n")
        f.write(f"Split strategy : 70% train | 15% validation | 15% test\n")
        f.write(f"N splits       : {N_SPLITS} (seeds {RANDOM_SEED}–{RANDOM_SEED+N_SPLITS-1})\n\n")
        f.write(f"=== Aggregate metrics (test set, mean ± std) ===\n")
        f.write(f"F1   : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}\n")
        f.write(f"AUC    : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}\n")
        f.write(f"AUC-PR : {summary['auc_pr_mean']:.4f} ± {summary['auc_pr_std']:.4f}\n")
        f.write(f"Prec (mining) : {summary['prec_mining_mean']:.4f} ± {summary['prec_mining_std']:.4f}\n")
        f.write(f"Rec  (mining) : {summary['rec_mining_mean']:.4f} ± {summary['rec_mining_std']:.4f}\n")
        f.write(f"Prec (resto)  : {summary['prec_resto_mean']:.4f} ± {summary['prec_resto_std']:.4f}\n")
        f.write(f"Rec  (resto)  : {summary['rec_resto_mean']:.4f} ± {summary['rec_resto_std']:.4f}\n")
        f.write(f"\n=== Best split report (seed={best_split['seed']}, "
                f"τ*={best_split['threshold']:.2f}) ===\n\n")
        f.write(report)

    plot_roc(best_split["y_test"], best_split["y_prob_test"],
             state_safe, OUTPUT_DIR)
    plot_pr(best_split["y_test"], best_split["y_prob_test"],
            state_safe, OUTPUT_DIR)
    plot_threshold_curve(best_split["y_val"], best_split["y_prob_val"],
                         best_split["threshold"], state_safe, OUTPUT_DIR)
    plot_confusion(best_split["y_test"], best_split["y_pred_test"],
                   state_safe, OUTPUT_DIR)

    summary["_base_models"] = best_split["base_models"]
    summary["_meta_model"]  = best_split["meta_model"]
    summary["_features"]    = features
    summary["_log_cols"]    = log_cols

    return summary


# ============================================================
# CROSS-STATE GENERALIZATION (Pará → others)
# ============================================================

def run_generalization(df, para_result):
    print(f"\n{'='*60}")
    print("  CROSS-STATE GENERALIZATION (Pará → others)")
    print(f"{'='*60}")

    base_models = para_result["_base_models"]
    meta_model  = para_result["_meta_model"]
    features    = para_result["_features"]
    log_cols    = para_result["_log_cols"]

    lines = [
        "CROSS-STATE GENERALIZATION REPORT\n",
        "Model: Pará best split → other states (no retraining)\n",
        "Note: threshold optimized on full target-state data (no val set available)\n",
        "="*60 + "\n"
    ]
    gen_rows = []

    for state in ["AMAZONAS", "MATO GROSSO", "RORAIMA"]:
        df_s       = df[df["ESTADO"] == state].copy()
        feat_avail = [f for f in features if f in df_s.columns]
        df_test    = df_s[feat_avail + [TARGET]].dropna()

        if len(df_test) < 50:
            continue

        X_ext = log_transform(df_test[feat_avail].copy(), log_cols)
        y_ext = df_test[TARGET].map(LABEL_MAP)

        positives = (y_ext == 1).sum()
        if positives < 10:
            lines.append(f"\n{state}: too few positives ({positives}), skipped.\n")
            continue

        for f in features:
            if f not in X_ext.columns:
                X_ext[f] = 0.0
        X_ext = X_ext[features]

        meta_X     = get_meta_features(base_models, X_ext)
        y_prob_ext = meta_model.predict_proba(meta_X)[:, 1]

        best_t_ext, best_f1_ext = optimize_threshold(y_ext, y_prob_ext)
        auc_ext     = roc_auc_score(y_ext, y_prob_ext)
        auc_pr_ext  = average_precision_score(y_ext, y_prob_ext)
        y_pred_ext  = (y_prob_ext > best_t_ext).astype(int)
        prec_mining = precision_score(y_ext, y_pred_ext, pos_label=1, zero_division=0)
        rec_mining  = recall_score(y_ext, y_pred_ext, pos_label=1, zero_division=0)

        report_ext = classification_report(
            y_ext, y_pred_ext,
            target_names=["resto (0)", "ilegal_mining (1)"],
            digits=4
        )
        line = (f"\n{state}\n"
                f"  Samples: {len(y_ext)} | Positives: {positives}\n"
                f"  Threshold : {best_t_ext:.2f}\n"
                f"  F1        : {best_f1_ext:.4f}\n"
                f"  AUC-ROC   : {auc_ext:.4f}\n"
                f"  Prec(mining): {prec_mining:.4f}\n"
                f"  Rec (mining): {rec_mining:.4f}\n\n"
                f"{report_ext}")
        lines.append(line)
        print(line)

        gen_rows.append({
            "state":       state,
            "n_total":     len(y_ext),
            "n_positive":  int(positives),
            "threshold":   round(best_t_ext, 2),
            "f1":          round(best_f1_ext, 4),
            "auc":         round(auc_ext, 4),
            "auc_pr":      round(auc_pr_ext, 4),
            "prec_mining": round(prec_mining, 4),
            "rec_mining":  round(rec_mining, 4),
        })

    with open(os.path.join(OUTPUT_DIR, "generalization_report.txt"), "w") as f:
        f.writelines(lines)

    if gen_rows:
        pd.DataFrame(gen_rows).to_csv(
            os.path.join(OUTPUT_DIR, "generalization_summary.csv"), index=False
        )

    return gen_rows


# ============================================================
# ENTRY POINT
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="SASE — State-Aware Stacking Ensemble")
    p.add_argument("--data", type=str, default=DATA_PATH,
                   help="Path to input CSV (default: config.DATA_CSV)")
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                   help="Directory for outputs (default: RESULTS_DIR/sase_res020)")
    p.add_argument("--block-res", type=float, default=BLOCK_RES,
                   help="Spatial block grid resolution in degrees (paper: 0.2)")
    p.add_argument("--n-splits", type=int, default=N_SPLITS,
                   help="Number of independent spatial splits (paper: 5)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="Base random seed (paper: 42)")
    return p.parse_args()


def main():
    global DATA_PATH, OUTPUT_DIR, BLOCK_RES, N_SPLITS, RANDOM_SEED
    args = parse_args()
    DATA_PATH   = args.data
    OUTPUT_DIR  = args.output_dir
    BLOCK_RES   = args.block_res
    N_SPLITS    = args.n_splits
    RANDOM_SEED = args.seed

    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"Input file not found: {DATA_PATH}\n"
            "Set DATA_DIR or pass --data. See data/README.md to obtain the dataset."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")

    print(f"Deriving spatial blocks from '{GEO_COL}' at {BLOCK_RES}° "
          f"(~{BLOCK_RES*111:.0f} km)...")
    df[BLOCK_COL] = compute_blocks(df, GEO_COL, BLOCK_RES)
    n_bad = df[BLOCK_COL].isna().sum()
    if n_bad:
        print(f"  WARNING: {n_bad} geometries could not be parsed; rows dropped.")
        df = df[df[BLOCK_COL].notna()].copy()
    print(f"Split: 70% train | 15% validation | 15% test (grouped by spatial block)")
    print(f"N_SPLITS = {N_SPLITS}  (seeds {RANDOM_SEED}–{RANDOM_SEED+N_SPLITS-1})\n")

    summary_rows = []
    para_result  = None

    for state, features in STATE_FEATURES.items():
        result = run_state(
            state, df, features,
            use_xgb=STATE_USE_XGB[state],
            use_mlp=STATE_USE_MLP[state]
        )
        summary_rows.append({k: v for k, v in result.items()
                              if not k.startswith("_")})
        if state == "PARÁ":
            para_result = result

    # Global summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "sase_summary.csv"), index=False)

    print("\n\n=== SASE FINAL SUMMARY (mean ± std, test set) ===")
    for _, row in summary_df.iterrows():
        print(f"\n{row['state']}")
        print(f"  F1  : {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
        print(f"  AUC : {row['auc_mean']:.4f} ± {row['auc_std']:.4f}")
        print(f"  Prec(mining): {row['prec_mining_mean']:.4f} ± {row['prec_mining_std']:.4f}")
        print(f"  Rec (mining): {row['rec_mining_mean']:.4f} ± {row['rec_mining_std']:.4f}")

    if para_result:
        run_generalization(df, para_result)

    print(f"\n✅ All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
