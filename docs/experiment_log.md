# Experiment log

Narrative record of the experiments behind the paper. Scope here is the **SASE**
pipeline that produces the paper's results; several exploratory tracks (a Siamese
network for cross-state embeddings, raw SAR time-series fusion, an EfficientNet
CNN on SAR patches) were run during development but are **not part of the
published paper and are not included in this repository**.

## Problem

Binary attribution of validated deforestation alert polygons across four
Amazonian states (Pará, Mato Grosso, Amazonas, Roraima) from Sentinel-1 C-band
SAR (VV/VH):

- **Class 1** — `ilegal_mining` (illegal mining)
- **Class 0** — `resto` (agriculture and all other drivers)

Main challenge: severe class imbalance (~1:12 in Amazonas — 339 positives out of
4,312) combined with heterogeneous mining patterns across states.

## Method — SASE (State-Aware Stacking Ensemble)

- **Base learners**: Random Forest + XGBoost + Logistic Regression + MLP
- **Meta-learner**: Logistic Regression on out-of-fold base predictions
- **Validation**: `GroupShuffleSplit` (5 splits) over 0.2° spatial blocks to avoid
  spatial leakage; 70/15/15 train/val/test
- **Threshold**: τ* optimized on the validation set to maximize F1
- **Per-state feature sets**: `STATE_FEATURES` in `src/sase_spatial.py`

Reproduce with `python src/sase_spatial.py` → `results/sase_res020/`.

### Per-state results (test set, mean ± std over 5 splits)

| State        | F1              | AUC    | Prec. (mining) | Rec. (mining) |
|--------------|-----------------|--------|----------------|---------------|
| Pará         | 0.881 ± 0.006   | 0.943  | 0.899          | 0.864         |
| Mato Grosso  | 0.742 ± 0.080   | 0.956  | 0.674          | 0.840         |
| Amazonas     | 0.520 ± 0.089   | 0.866  | 0.517          | 0.538         |
| Roraima      | 0.478 ± 0.091   | 0.904  | 0.411          | 0.572         |

Pará is far more balanced (13,266 samples, 6,720 positives); Amazonas and Roraima
suffer from severe imbalance, which explains the F1 gap despite reasonable AUC
(the model ranks well but the threshold is harder to place).

### Cross-state generalization (Pará → others, no retraining)

| Target state | F1     | AUC    |
|--------------|--------|--------|
| Mato Grosso  | ~0.77  | ~0.96  |
| Amazonas     | ~0.47  | ~0.82  |
| Roraima      | ~0.40  | ~0.86  |

Transfer works reasonably for Mato Grosso but fails for Amazonas/Roraima — the SAR
signatures of mining are **regionally specific** (soil, vegetation, mining type,
and hydrological context differ by state). This is the paper's central result.
See `results/reference_res020/generalization_summary.csv`.

## Feature contribution (summary)

| Feature group        | Effect                                             |
|----------------------|----------------------------------------------------|
| GLCM texture         | Largest single gain — spatial pattern of mining backscatter |
| SAR backscatter stats| Backbone discriminative features (VV/VH/VV−VH)     |
| Terrain/river context| Geographic regularizer — reduces variance across spatial splits |

Feature separability per state (Cliff's Delta) is computed by
`src/feature_analysis.py` → `feature_separability.csv`.

## Reference outputs

`results/reference_res020/` holds the committed reference produced from the
published dataset with the default parameters (`BLOCK_RES=0.2`, `RANDOM_SEED=42`,
`N_SPLITS=5`): `sase_summary.csv`, per-state classification reports and curves,
and `generalization_summary.csv`.
