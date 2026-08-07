"""
Feature separability analysis — per-class KDE/histogram grids, pairs plots, and a
Cliff's Delta separability table (used for the feature-analysis figures/table in
the paper).

Reads config.DATA_CSV; writes figures and feature_separability.csv to
RESULTS_DIR/feature_analysis (override with --output-dir).

Usage
-----
  python src/feature_analysis.py
  python src/feature_analysis.py --output-dir /path/to/out
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
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_CSV, RESULTS_DIR

warnings.filterwarnings('ignore')

# MDPI proof (Fig. 5/6): render the axis offset as x10^n (mathtext), not '1e7'.
plt.rcParams['axes.formatter.use_mathtext'] = True

STATE_COL  = "ESTADO"
LABEL_COL  = "VPRESSAO"
CLASS_POS  = "ilegal_mining"
CLASS_NEG  = "resto"

STATE_NAME_MAP = {
    "PARÁ":        "Para",
    "MATO GROSSO": "MatoGrosso",
    "AMAZONAS":    "Amazonas",
    "RORAIMA":     "Roraima",
}

# Features per state (as in the paper).
STATE_FEATURES = {
    "PARÁ": [
        "VV_stdDev", "VH_stdDev",
        "VV_minus_VH_mean", "VV_minus_VH_p10", "VV_minus_VH_p90",
        "contrast_vv_mean", "contrast_vv_p10", "contrast_vv_p90",
        "diss_vv_mean", "diss_vv_p10", "diss_vv_p90",
        "corr_vv_mean", "corr_vv_p10",
        "contrast_vh_mean", "contrast_vh_p10", "contrast_vh_p90",
        "diss_vh_mean", "diss_vh_p10", "diss_vh_p90",
        "corr_vh_mean", "corr_vh_p10", "corr_vh_p90",
        "tpi",
    ],
    "MATO GROSSO": [
        "VV_stdDev", "VH_stdDev",
        "VV_minus_VH_mean", "VV_minus_VH_p10", "VV_minus_VH_p90",
        "contrast_vv_mean", "contrast_vv_p10", "contrast_vv_p90",
        "diss_vv_mean", "diss_vv_p10", "diss_vv_p90",
        "corr_vv_mean", "corr_vv_p10",
        "contrast_vh_mean", "contrast_vh_p10", "contrast_vh_p90",
        "diss_vh_mean", "diss_vh_p10", "diss_vh_p90",
        "corr_vh_mean", "corr_vh_p10", "corr_vh_p90",
    ],
    "AMAZONAS": [
        "VV_stdDev", "VH_stdDev",
        "contrast_vv_mean", "contrast_vv_p10", "contrast_vv_p90",
        "diss_vv_mean", "diss_vv_p10", "diss_vv_p90",
        "corr_vv_mean", "corr_vv_p90",
        "contrast_vh_mean", "diss_vh_mean", "diss_vh_p90",
        "corr_vh_mean", "corr_vh_p90",
        "slope", "roughness", "dist_river",
    ],
    "RORAIMA": [
        "VV_stdDev", "VH_stdDev",
        "contrast_vv_mean", "contrast_vv_p10", "contrast_vv_p90",
        "diss_vv_mean", "diss_vv_p10", "diss_vv_p90",
        "corr_vv_mean", "corr_vv_p10", "corr_vv_p90",
        "corr_vh_mean", "corr_vh_p10", "corr_vh_p90",
        "dist_river",
    ],
}

OUTPUT_DIR = str(RESULTS_DIR / "feature_analysis")


def cliffs_delta(a, b):
    """Signed Cliff's delta: delta = 2U/(n1*n2) - 1 using Mann-Whitney U."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    return (2 * U) / (n1 * n2) - 1


def delta_magnitude(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.330:
        return "small"
    elif ad < 0.474:
        return "medium"
    return "large"


def plot_hist_grid(df_state, features, state_label, state_safe, deltas, output_dir):
    n_feat = len(features)
    n_cols = 5
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 3.2))
    axes_flat = axes.flatten() if n_rows > 1 else np.array(axes).flatten()

    grp_pos = df_state[df_state[LABEL_COL] == CLASS_POS]
    grp_neg = df_state[df_state[LABEL_COL] == CLASS_NEG]

    for i, feat in enumerate(features):
        ax = axes_flat[i]
        vals_pos = grp_pos[feat].dropna().values
        vals_neg = grp_neg[feat].dropna().values
        ax.hist(vals_neg, bins=30, density=True, alpha=0.4, color="steelblue", label="resto")
        ax.hist(vals_pos, bins=30, density=True, alpha=0.4, color="darkorange", label=CLASS_POS)
        for vals, color in [(vals_neg, "steelblue"), (vals_pos, "darkorange")]:
            if len(vals) >= 2:
                try:
                    kde = stats.gaussian_kde(vals)
                    xr = np.linspace(vals.min(), vals.max(), 300)
                    ax.plot(xr, kde(xr), color=color, lw=1.8)
                except np.linalg.LinAlgError:
                    pass
        if len(vals_neg) > 0:
            ax.axvline(vals_neg.mean(), color="steelblue", linestyle="--", lw=1.2)
        if len(vals_pos) > 0:
            ax.axvline(vals_pos.mean(), color="darkorange", linestyle="--", lw=1.2)
        d = deltas.get(feat, np.nan)
        if not np.isnan(d):
            ax.text(0.97, 0.95,
                    # MDPI proof (Fig. 5): ASCII hyphen -> minus sign (U+2212)
                    f"δ = {d:.2f} ({delta_magnitude(d)})".replace("-", "\u2212"),
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7))
        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n_feat, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Feature distributions by class — {state_label}", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, f"hist_{state_safe}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pairs(df_state, top_features, state_label, state_safe, output_dir):
    df_plot = df_state[top_features + [LABEL_COL]].dropna().copy()
    palette = {"resto": "steelblue", CLASS_POS: "darkorange"}
    pg = sns.pairplot(df_plot, hue=LABEL_COL, vars=top_features, palette=palette,
                      diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10},
                      diag_kws={"alpha": 0.5})
    pg.figure.suptitle(f"Pairs plot — Top 6 features by Cliff's Delta — {state_label}",
                       y=1.02, fontsize=12)
    out = os.path.join(output_dir, f"pairs_{state_safe}.png")
    pg.figure.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(pg.figure)
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_CSV))
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Input file not found: {args.data}\nSee data/README.md.")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset …")
    df = pd.read_csv(args.data)
    print(f"  Shape: {df.shape}")

    all_rows = []
    for estado, state_safe in STATE_NAME_MAP.items():
        print(f"\n{'='*60}\n  STATE: {estado}  ({state_safe})\n{'='*60}")
        df_state = df[df[STATE_COL] == estado].copy()
        features = [f for f in STATE_FEATURES[estado] if f in df_state.columns]

        grp_pos = df_state[df_state[LABEL_COL] == CLASS_POS]
        grp_neg = df_state[df_state[LABEL_COL] == CLASS_NEG]
        print(f"  Samples: {len(df_state)}  |  {CLASS_POS}: {len(grp_pos)}  |  resto: {len(grp_neg)}")

        deltas = {}
        for feat in features:
            a = grp_pos[feat].dropna().values
            b = grp_neg[feat].dropna().values
            d = cliffs_delta(a, b)
            deltas[feat] = d
            all_rows.append({
                "state": state_safe, "feature": feat,
                "mean_mining": float(a.mean()) if len(a) else np.nan,
                "mean_resto": float(b.mean()) if len(b) else np.nan,
                "delta": d,
                "magnitude": delta_magnitude(d) if not np.isnan(d) else "n/a",
                "separable": bool(abs(d) >= 0.474) if not np.isnan(d) else False,
            })

        plot_hist_grid(df_state, features, estado, state_safe, deltas, args.output_dir)
        top6 = sorted(features, key=lambda f: abs(deltas[f]), reverse=True)[:6]
        print(f"  Top-6 by |δ|: {top6}")
        plot_pairs(df_state, top6, estado, state_safe, args.output_dir)

    sep_df = pd.DataFrame(all_rows).sort_values(
        ["state", "delta"],
        key=lambda c: c.abs() if c.name == "delta" else c,
        ascending=[True, False])
    csv_path = os.path.join(args.output_dir, "feature_separability.csv")
    sep_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
