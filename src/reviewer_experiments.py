"""
reviewer_experiments.py
=======================
Reviewer-requested experiments for the revision, run in the SAME pinned
environment as ``sase_spatial.py`` so that every reported number is mutually
consistent with the per-state results in Table ``tab:sase_performance`` (the
canonical run reproduced bit-for-bit by ``sase_spatial.py --block-res 0.2``).

All building blocks (base learners, leak-free OOF stacking, spatial split,
threshold optimisation) are imported from ``sase_spatial`` — this script only
orchestrates the additional experiments, so the ensemble itself is identical.

Experiments
-----------
  base_vs_stack   (R2.1)  individual base learners vs the SASE ensemble
  fusion          (R2.4)  matched (MT+PA->MT) and mismatched (PA+MT->AM/RR)
  imbalance       (R2.5)  SMOTE / random-over-sampling ablation (AM, RR; RF, XGB)
  threshold_wide  (R4.7)  decision-threshold grid widened to [0.2, 0.95]
  intersection    (R4.3)  Para->AM/RR using only the features common to all states
  std_suffix      (R4.1)  add back the excluded stdDev GLCM features
  ks_test         (R3.4)  Kolmogorov-Smirnov corroboration of the MW/FDR selection

  (block-size sensitivity is produced by re-running sase_spatial.py at
   --block-res 0.1 and 0.3; the feature-importance ranking by feature_analysis.py)

Usage
-----
  python src/reviewer_experiments.py                 # run all
  python src/reviewer_experiments.py --only fusion   # run one
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score,
)
from scipy import stats
from imblearn.over_sampling import SMOTE, RandomOverSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the exact SASE machinery.
import sase_spatial as S
from sase_spatial import (
    build_base_models, get_oof_meta_features, get_meta_features,
    compute_blocks, log_transform, get_log_cols,
    STATE_FEATURES, LABEL_MAP,
    TARGET, GROUP_COL, BLOCK_COL, GEO_COL,
    VAL_TEST_SIZE, TEST_FROM_VT, RANDOM_SEED, N_SPLITS, N_META_FOLDS,
)
from config import DATA_CSV, RESULTS_DIR

warnings.filterwarnings("ignore")

OUT_DIR = str(RESULTS_DIR / "reviewer_experiments")
STATES  = ["PARÁ", "MATO GROSSO", "AMAZONAS", "RORAIMA"]

# The six excluded stdDev GLCM features that DO exist in the dataset (ord_flow is
# absent from the modelling CSV and is therefore not part of this ablation).
STD_SUFFIX_FEATURES = [
    "contrast_vv_stdDev", "diss_vv_stdDev", "corr_vv_stdDev",
    "contrast_vh_stdDev", "diss_vh_stdDev", "corr_vh_stdDev",
]


# ---------------------------------------------------------------------------
# shared helpers (identical split + SASE fit/eval as sase_spatial)
# ---------------------------------------------------------------------------
def spatial_partitions(X, y, groups, seed):
    """70/15/15 spatial split, identical to sase_spatial.run_single_split."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=VAL_TEST_SIZE, random_state=seed)
    tr, vt = next(gss1.split(X, y, groups))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], groups.iloc[tr]
    Xvt, yvt, gvt = X.iloc[vt], y.iloc[vt], groups.iloc[vt]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=TEST_FROM_VT, random_state=seed + 100)
    va, te = next(gss2.split(Xvt, yvt, gvt))
    return {
        "Xtr": Xtr, "ytr": ytr, "gtr": gtr,
        "Xva": Xvt.iloc[va], "yva": yvt.iloc[va],
        "Xte": Xvt.iloc[te], "yte": yvt.iloc[te],
    }


def opt_threshold(y_true, y_prob, grid):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (y_prob > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def fit_sase(Xtr, ytr, gtr, seed, use_xgb=True, use_mlp=True):
    base = build_base_models(ytr, use_xgb=use_xgb, use_mlp=use_mlp)
    meta_tr = get_oof_meta_features(base, Xtr, ytr, gtr, n_folds=N_META_FOLDS, seed=seed)
    for m in base.values():
        m.fit(Xtr, ytr)
    meta = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
    meta.fit(meta_tr, ytr)
    return base, meta


def sase_probs(base, meta, X):
    return meta.predict_proba(get_meta_features(base, X))[:, 1]


def state_frame(df, state, features):
    """Model matrix for one state (dropna, label map, log-transform)."""
    feats = [f for f in features if f in df.columns]
    d = df[df["ESTADO"] == state][feats + [TARGET, GROUP_COL, BLOCK_COL]].dropna()
    X = log_transform(d[feats].copy(), get_log_cols(feats))
    y = d[TARGET].map(LABEL_MAP)
    g = d[BLOCK_COL]
    return X, y, g, feats


def metrics(y_true, y_prob, thr):
    y_pred = (y_prob > thr).astype(int)
    return {
        "f1":     f1_score(y_true, y_pred, zero_division=0),
        "auc":    roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "prec":   precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "rec":    recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def mean_std(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)), float(np.std(v))


def bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR: boolean mask of rejected (significant) hypotheses."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, dtype=bool)
    if k > 0:
        rej[order[:k]] = True
    return rej


# ---------------------------------------------------------------------------
# R2.1 — base learners vs SASE
# ---------------------------------------------------------------------------
def exp_base_vs_stack(df):
    print("\n=== base_vs_stack (R2.1) ===")
    grid = S.THRESHOLD_RANGE
    out = []
    for state in STATES:
        X, y, g, feats = state_frame(df, state, STATE_FEATURES[state])
        per_model = {m: [] for m in ["RF", "XGB", "LR", "MLP", "SASE"]}
        for i in range(N_SPLITS):
            seed = RANDOM_SEED + i
            p = spatial_partitions(X, y, g, seed)
            base, meta = fit_sase(p["Xtr"], p["ytr"], p["gtr"], seed)
            # individual base learners (own optimal threshold on val)
            for name, model in base.items():
                pv = model.predict_proba(p["Xva"])[:, 1]
                pt = model.predict_proba(p["Xte"])[:, 1]
                t, _ = opt_threshold(p["yva"], pv, grid)
                per_model[name].append(metrics(p["yte"], pt, t))
            # SASE
            pv = sase_probs(base, meta, p["Xva"])
            pt = sase_probs(base, meta, p["Xte"])
            t, _ = opt_threshold(p["yva"], pv, grid)
            per_model["SASE"].append(metrics(p["yte"], pt, t))
        for m in ["RF", "XGB", "LR", "MLP", "SASE"]:
            f1m, f1s = mean_std(per_model[m], "f1")
            prm, _   = mean_std(per_model[m], "auc_pr")
            aum, _   = mean_std(per_model[m], "auc")
            out.append({"state": state, "model": m,
                        "f1": round(f1m, 4), "auc": round(aum, 4),
                        "auc_pr": round(prm, 4)})
            print(f"  {state:12s} {m:5s} F1={f1m:.4f} AUC-PR={prm:.4f}")
    pd.DataFrame(out).to_csv(os.path.join(OUT_DIR, "base_vs_stack.csv"), index=False)


# ---------------------------------------------------------------------------
# R2.4 — data fusion (pooling)
# ---------------------------------------------------------------------------
def exp_fusion(df):
    print("\n=== fusion (R2.4) ===")
    grid = S.THRESHOLD_RANGE
    rows = []

    # ---- Matched contrast: MT training augmented with all Pará (eval MT test)
    feats = [f for f in STATE_FEATURES["MATO GROSSO"] if f in df.columns]
    Xmt, ymt, gmt, _ = state_frame(df, "MATO GROSSO", feats)
    # Pará rows expressed in MT's feature space (all these features are global)
    dpa = df[df["ESTADO"] == "PARÁ"][feats + [TARGET, BLOCK_COL]].dropna()
    Xpa = log_transform(dpa[feats].copy(), get_log_cols(feats))
    ypa = dpa[TARGET].map(LABEL_MAP)

    local, pooled = [], []
    for i in range(N_SPLITS):
        seed = RANDOM_SEED + i
        p = spatial_partitions(Xmt, ymt, gmt, seed)
        # local baseline (MT only)
        b, m = fit_sase(p["Xtr"], p["ytr"], p["gtr"], seed)
        t, _ = opt_threshold(p["yva"], sase_probs(b, m, p["Xva"]), grid)
        local.append(metrics(p["yte"], sase_probs(b, m, p["Xte"]), t))
        # pooled (MT train + all Pará). Spatial groups: MT blocks + a distinct
        # Pará block id so grouped OOF still avoids leakage across sources.
        Xtr2 = pd.concat([p["Xtr"], Xpa], ignore_index=True)
        ytr2 = pd.concat([p["ytr"], ypa], ignore_index=True)
        gtr2 = pd.concat([p["gtr"].astype(str),
                          pd.Series("PA_" + dpa[BLOCK_COL].astype(str).values)],
                         ignore_index=True)
        b, m = fit_sase(Xtr2, ytr2, gtr2, seed)
        t, _ = opt_threshold(p["yva"], sase_probs(b, m, p["Xva"]), grid)
        pooled.append(metrics(p["yte"], sase_probs(b, m, p["Xte"]), t))
    for label, r in [("MT local", local), ("PA+MT pooled", pooled)]:
        rows.append({"setting": "matched", "model": label,
                     "f1": round(mean_std(r, "f1")[0], 4),
                     "auc": round(mean_std(r, "auc")[0], 4),
                     "auc_pr": round(mean_std(r, "auc_pr")[0], 4),
                     "prec": round(mean_std(r, "prec")[0], 4),
                     "rec": round(mean_std(r, "rec")[0], 4)})
        print(f"  matched  {label:14s} "
              f"F1={rows[-1]['f1']} AUC-PR={rows[-1]['auc_pr']} P={rows[-1]['prec']}")

    # ---- Mismatched contrast: pooled PA+MT model applied to AM, RR (no retrain)
    src_feats = [f for f in STATE_FEATURES["PARÁ"] if f in df.columns]
    dsrc = df[df["ESTADO"].isin(["PARÁ", "MATO GROSSO"])][src_feats + [TARGET, BLOCK_COL]].dropna()
    Xsrc = log_transform(dsrc[src_feats].copy(), get_log_cols(src_feats))
    ysrc = dsrc[TARGET].map(LABEL_MAP)
    gsrc = dsrc[BLOCK_COL]
    # best-split source model (mirrors the generalization protocol in the paper)
    best = None
    for i in range(N_SPLITS):
        seed = RANDOM_SEED + i
        p = spatial_partitions(Xsrc, ysrc, gsrc, seed)
        b, m = fit_sase(p["Xtr"], p["ytr"], p["gtr"], seed)
        t, _ = opt_threshold(p["yva"], sase_probs(b, m, p["Xva"]), grid)
        f1 = metrics(p["yte"], sase_probs(b, m, p["Xte"]), t)["f1"]
        if best is None or f1 > best[0]:
            best = (f1, b, m)
    _, bpool, mpool = best
    for tgt in ["AMAZONAS", "RORAIMA"]:
        dt = df[df["ESTADO"] == tgt][[c for c in src_feats if c in df.columns] + [TARGET]].dropna()
        Xt = log_transform(dt[[c for c in src_feats if c in df.columns]].copy(),
                            get_log_cols(src_feats))
        for f in src_feats:
            if f not in Xt.columns:
                Xt[f] = 0.0
        Xt = Xt[src_feats]
        yt = dt[TARGET].map(LABEL_MAP)
        pr = sase_probs(bpool, mpool, Xt)
        t, _ = opt_threshold(yt, pr, grid)
        mm = metrics(yt, pr, t)
        rows.append({"setting": "mismatched", "model": tgt,
                     "f1": round(mm["f1"], 4), "auc": round(mm["auc"], 4),
                     "auc_pr": round(mm["auc_pr"], 4),
                     "prec": round(mm["prec"], 4), "rec": round(mm["rec"], 4)})
        print(f"  mismatched {tgt:9s} pooled F1={mm['f1']:.4f} AUC={mm['auc']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "fusion.csv"), index=False)


# ---------------------------------------------------------------------------
# R2.5 — imbalance mitigation ablation (AM, RR; RF, XGB)
# ---------------------------------------------------------------------------
def exp_imbalance(df):
    print("\n=== imbalance (R2.5) ===")
    grid = S.THRESHOLD_RANGE
    out = []
    for state in ["AMAZONAS", "RORAIMA"]:
        X, y, g, feats = state_frame(df, state, STATE_FEATURES[state])
        for strat in ["baseline", "SMOTE", "random over-sampling"]:
            res = {"RF": [], "XGB": []}
            for i in range(N_SPLITS):
                seed = RANDOM_SEED + i
                p = spatial_partitions(X, y, g, seed)
                Xtr, ytr = p["Xtr"], p["ytr"]
                if strat == "SMOTE":
                    k = max(1, min(5, int((ytr == 1).sum()) - 1))
                    Xtr, ytr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(Xtr, ytr)
                elif strat == "random over-sampling":
                    Xtr, ytr = RandomOverSampler(random_state=seed).fit_resample(Xtr, ytr)
                # for resampled sets, class balance is handled by resampling ->
                # plain learners; baseline keeps the class-weight learners.
                use_weight = (strat == "baseline")
                learners = build_base_models(ytr, use_xgb=True, use_mlp=False)
                if not use_weight:
                    from sklearn.ensemble import RandomForestClassifier
                    from xgboost import XGBClassifier
                    learners = {
                        "RF": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                     random_state=RANDOM_SEED),
                        "XGB": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                             eval_metric="logloss", random_state=RANDOM_SEED),
                    }
                for name in ["RF", "XGB"]:
                    mdl = learners[name]
                    mdl.fit(Xtr, ytr)
                    pv = mdl.predict_proba(p["Xva"])[:, 1]
                    pt = mdl.predict_proba(p["Xte"])[:, 1]
                    t, _ = opt_threshold(p["yva"], pv, grid)
                    res[name].append(metrics(p["yte"], pt, t))
            row = {"state": state, "strategy": strat}
            for name in ["RF", "XGB"]:
                row[f"{name}_f1"] = round(mean_std(res[name], "f1")[0], 4)
                row[f"{name}_auc_pr"] = round(mean_std(res[name], "auc_pr")[0], 4)
            out.append(row)
            print(f"  {state:9s} {strat:20s} "
                  f"RF F1={row['RF_f1']} XGB F1={row['XGB_f1']}")
    pd.DataFrame(out).to_csv(os.path.join(OUT_DIR, "imbalance.csv"), index=False)


# ---------------------------------------------------------------------------
# R4.7 — widened threshold grid [0.2, 0.95]
# ---------------------------------------------------------------------------
def exp_threshold_wide(df):
    print("\n=== threshold_wide (R4.7) ===")
    narrow = S.THRESHOLD_RANGE
    wide = np.linspace(0.2, 0.95, 76)
    out = []
    for state in ["AMAZONAS", "RORAIMA"]:
        X, y, g, feats = state_frame(df, state, STATE_FEATURES[state])
        r_n, r_w, tsat = [], [], []
        for i in range(N_SPLITS):
            seed = RANDOM_SEED + i
            p = spatial_partitions(X, y, g, seed)
            b, m = fit_sase(p["Xtr"], p["ytr"], p["gtr"], seed)
            pv = sase_probs(b, m, p["Xva"])
            pt = sase_probs(b, m, p["Xte"])
            tn, _ = opt_threshold(p["yva"], pv, narrow)
            tw, _ = opt_threshold(p["yva"], pv, wide)
            r_n.append(f1_score(p["yte"], (pt > tn).astype(int), zero_division=0))
            r_w.append(f1_score(p["yte"], (pt > tw).astype(int), zero_division=0))
            tsat.append(tw)
        out.append({"state": state,
                    "f1_narrow_0.2_0.8": round(float(np.mean(r_n)), 4),
                    "f1_wide_0.2_0.95": round(float(np.mean(r_w)), 4),
                    "mean_wide_threshold": round(float(np.mean(tsat)), 3)})
        print(f"  {state:9s} narrow={out[-1]['f1_narrow_0.2_0.8']} "
              f"wide={out[-1]['f1_wide_0.2_0.95']} (tau~{out[-1]['mean_wide_threshold']})")
    pd.DataFrame(out).to_csv(os.path.join(OUT_DIR, "threshold_wide.csv"), index=False)


# ---------------------------------------------------------------------------
# R4.3 — intersection of features common to all four states (no imputation)
# ---------------------------------------------------------------------------
def exp_intersection(df):
    print("\n=== intersection (R4.3) ===")
    common = set(STATE_FEATURES["PARÁ"])
    for s in STATES[1:]:
        common &= set(STATE_FEATURES[s])
    common = [f for f in STATE_FEATURES["PARÁ"] if f in common]  # stable order
    print(f"  common features ({len(common)}): {common}")
    grid = S.THRESHOLD_RANGE

    Xp, yp, gp, _ = state_frame(df, "PARÁ", common)
    best = None
    for i in range(N_SPLITS):
        seed = RANDOM_SEED + i
        p = spatial_partitions(Xp, yp, gp, seed)
        b, m = fit_sase(p["Xtr"], p["ytr"], p["gtr"], seed)
        t, _ = opt_threshold(p["yva"], sase_probs(b, m, p["Xva"]), grid)
        f1 = f1_score(p["yte"], (sase_probs(b, m, p["Xte"]) > t).astype(int), zero_division=0)
        if best is None or f1 > best[0]:
            best = (f1, b, m)
    _, b, m = best
    out = [{"n_common_features": len(common), "features": ";".join(common)}]
    for tgt in ["AMAZONAS", "RORAIMA"]:
        Xt, yt, _, _ = state_frame(df, tgt, common)  # all common feats present -> no imputation
        pr = sase_probs(b, m, Xt)
        t, _ = opt_threshold(yt, pr, grid)
        mm = metrics(yt, pr, t)
        out.append({"target": tgt, "f1": round(mm["f1"], 4), "auc": round(mm["auc"], 4)})
        print(f"  Pará->{tgt:9s} F1={mm['f1']:.4f} AUC={mm['auc']:.4f}")
    pd.DataFrame(out).to_csv(os.path.join(OUT_DIR, "intersection.csv"), index=False)


# ---------------------------------------------------------------------------
# R4.1 — add back the excluded stdDev GLCM features
# ---------------------------------------------------------------------------
def exp_std_suffix(df):
    print("\n=== std_suffix (R4.1) ===")
    grid = S.THRESHOLD_RANGE
    out = []
    for state in STATES:
        base_feats = [f for f in STATE_FEATURES[state] if f in df.columns]
        add = [f for f in STD_SUFFIX_FEATURES if f in df.columns and f not in base_feats]
        for label, feats in [("selected", base_feats), ("selected+std", base_feats + add)]:
            X, y, g, _ = state_frame(df, state, feats)
            f1s = []
            for i in range(N_SPLITS):
                seed = RANDOM_SEED + i
                p = spatial_partitions(X, y, g, seed)
                b, m = fit_sase(p["Xtr"], p["ytr"], p["gtr"], seed)
                t, _ = opt_threshold(p["yva"], sase_probs(b, m, p["Xva"]), grid)
                f1s.append(f1_score(p["yte"], (sase_probs(b, m, p["Xte"]) > t).astype(int),
                                    zero_division=0))
            out.append({"state": state, "feature_set": label,
                        "n_features": len(feats),
                        "f1": round(float(np.mean(f1s)), 4),
                        "f1_std": round(float(np.std(f1s)), 4)})
            print(f"  {state:12s} {label:14s} ({len(feats)} feats) F1={out[-1]['f1']:.4f}")
    pd.DataFrame(out).to_csv(os.path.join(OUT_DIR, "std_suffix.csv"), index=False)


# ---------------------------------------------------------------------------
# R3.4 — Kolmogorov-Smirnov corroboration of the MW/FDR selection
# ---------------------------------------------------------------------------
def exp_ks_test(df):
    print("\n=== ks_test (R3.4) ===")
    # Candidate pool = all numeric feature columns actually used across states.
    non_feat = {TARGET, GROUP_COL, GEO_COL, "ESTADO", BLOCK_COL}
    candidates = [c for c in df.columns
                  if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])]
    out = []
    lead = {}
    for state in STATES:
        d = df[df["ESTADO"] == state]
        pos = d[d[TARGET] == "ilegal_mining"]
        neg = d[d[TARGET] == "resto"]
        mw_p, ks_p, ks_d, names = [], [], [], []
        for f in candidates:
            a = pos[f].dropna().values
            b = neg[f].dropna().values
            if len(a) < 3 or len(b) < 3:
                continue
            names.append(f)
            mw_p.append(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
            ks = stats.ks_2samp(a, b)
            ks_p.append(ks.pvalue)
            ks_d.append(ks.statistic)
        mw_p = np.array(mw_p); ks_p = np.array(ks_p)
        mw_sig = bh_reject(mw_p, alpha=0.05)
        ks_sig = bh_reject(ks_p, alpha=0.05)
        n_mw = int(mw_sig.sum())
        n_both = int((mw_sig & ks_sig).sum())
        out.append({"state": state, "n_candidates": len(names),
                    "n_MW_FDR_sig": n_mw,
                    "n_MW_and_KS_sig": n_both})
        # KS D for the leading texture feature
        if "contrast_vv_mean" in names:
            lead[state] = round(ks_d[names.index("contrast_vv_mean")], 3)
        print(f"  {state:12s} candidates={len(names)} MW-FDR sig={n_mw} "
              f"also KS sig={n_both} | contrast_vv_mean KS D={lead.get(state)}")
    pd.DataFrame(out).to_csv(os.path.join(OUT_DIR, "ks_test.csv"), index=False)
    pd.DataFrame([{"state": k, "contrast_vv_mean_KS_D": v} for k, v in lead.items()]) \
        .to_csv(os.path.join(OUT_DIR, "ks_leading_feature.csv"), index=False)


EXPERIMENTS = {
    "base_vs_stack":  exp_base_vs_stack,
    "fusion":         exp_fusion,
    "imbalance":      exp_imbalance,
    "threshold_wide": exp_threshold_wide,
    "intersection":   exp_intersection,
    "std_suffix":     exp_std_suffix,
    "ks_test":        exp_ks_test,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(EXPERIMENTS), default=None,
                    help="run a single experiment (default: all)")
    ap.add_argument("--block-res", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading dataset...")
    df = pd.read_csv(DATA_CSV)
    S.BLOCK_RES = args.block_res
    df[BLOCK_COL] = compute_blocks(df, GEO_COL, args.block_res)
    df = df[df[BLOCK_COL].notna()].copy()
    print(f"Shape: {df.shape} | blocks @ {args.block_res}°")

    todo = [args.only] if args.only else list(EXPERIMENTS)
    for name in todo:
        EXPERIMENTS[name](df)
    print(f"\n✅ Reviewer experiments written to {OUT_DIR}")


if __name__ == "__main__":
    main()
