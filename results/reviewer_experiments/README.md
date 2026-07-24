# Reviewer-round experiments — reproduction guide

All experiments in this folder were produced by
[`src/reviewer_experiments.py`](../../src/reviewer_experiments.py) in the **pinned
environment** (`requirements.txt`), which reproduces the per-state SASE metrics of
the main results table (`results/reference_res020/sase_summary.csv`) **bit-for-bit**.
Because every experiment reuses the exact SASE machinery of
[`src/sase_spatial.py`](../../src/sase_spatial.py) in that same environment, all the
numbers here are mutually consistent with the paper's Table "SASE performance per
state" (e.g. the `SASE` column of `base_vs_stack.csv` and the `selected` rows of
`std_suffix.csv` equal the per-state F1 of that table exactly).

## Setup (once)

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
# macOS without Homebrew: let XGBoost find libomp via scikit-learn's copy
export DYLD_FALLBACK_LIBRARY_PATH=$(./.venv/bin/python -c \
  "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))")
```

## Run

```bash
python src/reviewer_experiments.py               # all experiments
python src/reviewer_experiments.py --only fusion  # a single experiment
```

## Experiment → reviewer comment → output → manuscript location

| Experiment (`--only`) | Reviewer | Output CSV | Backs (manuscript) |
|---|---|---|---|
| `base_vs_stack`  | R2.1 | `base_vs_stack.csv` | Table *Individual base learners vs SASE* (`tab:base_vs_stack`) |
| `fusion`         | R2.4 | `fusion.csv` | Table *Data-fusion experiments* (`tab:fusion`) |
| `imbalance`      | R2.5 | `imbalance.csv` | Table *Imbalance-mitigation ablation* (`tab:imbalance`) |
| `threshold_wide` | R4.7 | `threshold_wide.csv` | §Limitations (threshold grid $[0.2,0.95]$) |
| `intersection`   | R4.3 | `intersection.csv` | §Limitations (11 common features, no imputation) |
| `std_suffix`     | R4.1 | `std_suffix.csv` | §Feature Selection (add-back of excluded `stdDev` features) |
| `ks_test`        | R3.4 | `ks_test.csv`, `ks_leading_feature.csv` | §Feature Selection (KS corroboration) |

Block-size sensitivity (R4.2, Table `tab:blocksize`) is produced separately by
re-running the main pipeline at three resolutions; the per-resolution summaries are
saved here for convenience:

```bash
python src/sase_spatial.py --block-res 0.1 --output-dir results/sase_res010
python src/sase_spatial.py --block-res 0.2 --output-dir results/sase_res020   # == reference
python src/sase_spatial.py --block-res 0.3 --output-dir results/sase_res030
```
→ `blocksize_res010_summary.csv`, `blocksize_res020_summary.csv`, `blocksize_res030_summary.csv`.

The feature-importance ranking (R2.8, Table `tab:feature_importance`) and the Cliff's
δ separability values come from `src/feature_analysis.py`
(`results/feature_analysis/feature_separability.csv`); these are data-derived and do
not depend on the modelling environment.

## Consistency check (canonical anchor)

Per-state SASE F1 at the paper's 0.2° blocks — identical across all sources:

| | Pará | Mato Grosso | Amazonas | Roraima |
|---|---|---|---|---|
| `reference_res020` (Table 6) | 0.881 | 0.742 | 0.520 | 0.478 |
| `base_vs_stack.csv` (SASE)   | 0.881 | 0.742 | 0.520 | 0.478 |
| `std_suffix.csv` (selected)  | 0.881 | 0.742 | 0.520 | 0.478 |
