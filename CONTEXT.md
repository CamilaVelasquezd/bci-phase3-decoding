# CONTEXT.md — bci-phase3-decoding

**Intern:** Camila Velásquez Díaz  
**Supervisor:** Juan Pablo Botero  
**Lab:** Solzbacher Lab, University of Utah  
**Project:** Phase 3 — Classical Decoding & Machine Learning for BCI Motor Decoding  

---

## 1. What this repo is and where it fits

This repo (`bci-phase3-decoding`) is the **Phase 3 decoding module** of the Solzbacher Lab BCI internship.  
It is intentionally **separate** from `bci-decoding-dataset` (the ingestion + visualization repo).

The separation reflects the lab's future architecture:
- `bci-decoding-dataset` → dataset ingestion pipelines and `neuroviz` visualization package only
- `bci-phase3-decoding` (this repo) → feature extraction, dimensionality reduction, and decoding models

Data always comes from S3 via the `bci_decoding_dataset` package — this repo does **not** re-implement ingestion.

---

## 2. Environment

| Item | Value |
|---|---|
| Conda env | `bci-ds` (Python 3.10, Windows) |
| AWS profile | `cv-pc` stored in `~/.aws/credentials` |
| S3 bucket | `solzbacher-lab-motor-decoding-ds` |
| S3 path | `datasets/Combined_Motor_Datasets` |
| Key packages | sklearn 1.7.2, numpy 1.26.4, scipy 1.15.3, seaborn 0.13.2 |
| Other packages | zarr, s3fs, xarray, dask, matplotlib, plotly |

The `bci-ds` environment is shared with `bci-decoding-dataset` — no new environment needed.  
If a new package is required, `pip install <pkg> --break-system-packages` inside `bci-ds`.

---

## 3. The four datasets

| Dataset | `dataset_id` | `task_type` | Electrodes | Notes |
|---|---|---|---|---|
| DANDI:000070 | `DANDI_00070` | `center_out` | ~191 | ~3000 trials; **best for discrete decoding** |
| DANDI:000688 | `DANDI_000688` | `center_out` | ~54 | ~180 trials per session |
| Zenodo:3854034 | `Zenodo_3854034` | `continuous_random` | ~96 | No discrete trials — `trial_phase` always 2; **skip for discrete decoding** |
| DANDI:000140 | `DANDI_000140` | `center_out_maze` | — | **KNOWN BUG: vx and vy columns are duplicated (same values). Do NOT use `velocity` from this dataset as regression target.** Use position or skip it. |

---

## 4. Data schema (V8)

Every session is loaded as `ds = data_utils.get_processed_data_from_session(session_id)` → `xr.Dataset`:

| Variable | Shape | Dtype | Notes |
|---|---|---|---|
| `spikes` | `(n_electrodes, n_time)` | uint8 | **Rows = electrodes, cols = time** — must transpose for sklearn |
| `position` | `(n_time, 2)` | float64 | Normalized x,y cursor position |
| `velocity` | `(n_time, 2)` | float64 | Normalized vx,vy — do NOT use from DANDI_000140 |
| `trial_id` | `(n_time,)` | int16 | 0=inter-trial, +N=successful trial N, -N=failed trial N |
| `trial_phase` | `(n_time,)` | int8 | 0=inter, 1=pre-reach, 2=reach, 3=post-reach |

Session attributes: `ds.attrs['task_type']`, `ds.attrs['dataset_id']`, `ds.attrs['sampling_rate']` (always 1000.0 Hz).

---

## 5. S3 connection pattern (copy exactly)

```python
import configparser
import os
import zarr
import s3fs
from bci_decoding_dataset.combined_dataset_utils import Combined_Dataset_Utils as d_utils

credentials_path = os.path.expanduser("~/.aws/credentials")
config = configparser.ConfigParser()
config.read(credentials_path)
profile = os.environ.get("AWS_PROFILE", "cv-pc")

dataset_kwargs = {
    "aws_store": True,
    "s3_bucket": "solzbacher-lab-motor-decoding-ds",
    "s3_key": "datasets/Combined_Motor_Datasets",
    "aws_access_key_id": config[profile]["aws_access_key_id"],
    "aws_secret_access_key": config[profile]["aws_secret_access_key"]
}
data_utils = d_utils(**dataset_kwargs)

# Direct S3 — needed for listing sessions
s3 = s3fs.S3FileSystem(
    key=config[profile]["aws_access_key_id"],
    secret=config[profile]["aws_secret_access_key"],
    client_kwargs={"region_name": "us-east-1"}
)
s3_store = s3fs.S3Map(
    root="s3://solzbacher-lab-motor-decoding-ds/datasets/Combined_Motor_Datasets",
    s3=s3, check=False
)
root = zarr.open_group(s3_store, mode="r")

# List and load sessions
dandi_70 = data_utils.filter_sessions(filter_by="dataset_id", filter_value="DANDI_00070")
session_id = dandi_70[0]
ds = data_utils.get_processed_data_from_session(session_id)
```

**WARNING:** `data_utils.combined_zarr.keys()` returns empty on S3. Always list sessions via `root.group_keys()` or `data_utils.filter_sessions()`.

---

## 6. Data access pattern

```python
# Load into memory (force eager evaluation from dask)
spikes      = ds["spikes"].values         # (n_electrodes, n_time), uint8
position    = ds["position"].values       # (n_time, 2), float64
velocity    = ds["velocity"].values       # (n_time, 2), float64
trial_id    = ds["trial_id"].values       # (n_time,), int16
trial_phase = ds["trial_phase"].values    # (n_time,), int8

# Trial masks
active_mask          = trial_id != 0          # excludes inter-trial periods
successful_mask      = trial_id > 0           # successful trials only
successful_trial_ids = np.unique(trial_id[trial_id > 0])
```

---

## 7. Phase 3 — Scientific plan

The decoding pipeline follows this sequence:

```
Spikes → Feature extraction → Dimensionality reduction (unsupervised) → Supervised model → Output
```

### 7.1 Feature extraction (input to all models)

- Bin spike trains into count matrices: `(n_bins, n_electrodes)` at 50ms bins
- Optional: Gaussian smoothing over time axis
- Average per trial for trial-level features: `(n_trials, n_electrodes)`

### 7.2 Dimensionality reduction (unsupervised — applied before supervised models)

Start with these, in order:

| Method | Library | Notes |
|---|---|---|
| **PCA** | `sklearn.decomposition.PCA` | Start here — fastest, most interpretable; produce scree plot |
| **ICA** | `sklearn.decomposition.FastICA` | Explores statistically independent components |
| **LDA** | `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | Also a classifier; as dim-reduction it finds directions that separate classes |
| Factor Analysis | `sklearn.decomposition.FactorAnalysis` | Optional later |

For PCA: always produce a **scree plot** and a **PC1 vs PC2 scatter colored by trial phase or direction**.

### 7.3 Regression (continuous decoding → vx/vy)

```
Spikes → dim reduction → Supervised regression → vx/vy velocity
```

| Model | Notes |
|---|---|
| **Wiener filter** | Linear regression with time lags; classical BCI baseline |
| **Ridge regression** | Regularized linear regression; good first benchmark |
| **Kalman filter** | Probabilistic; models dynamics explicitly — implement after linear baselines |

Metric: R² score per velocity dimension (vx and vy separately).  
Use `shuffle=False` in all train/test splits.

### 7.4 Classification (discrete decoding → reach direction)

```
Spikes → dim reduction → Supervised classification → direction (1 of 8)
```

| Model | Notes |
|---|---|
| **RNN (LSTM)** | Sequence model; handles temporal dynamics |
| **RNN (GRUD)** | GRU with decay for missing data |

Metric: accuracy across 8 directions.  
Only use `task_type in ('center_out', 'center_out_maze')` datasets.  
Preferred dataset for classification: `DANDI_00070` (~3000 trials, 8 directions, 191 electrodes).

---

## 8. Repo structure (target)

```
bci-phase3-decoding/
├── CONTEXT.md                          ← this file
├── README.md
├── requirements.txt
├── decoding/
│   ├── __init__.py                     ← exports DecodingPipeline + feature functions
│   ├── feature_extraction.py           ← compute_binned_counts, smooth_firing_rates,
│   │                                      compute_trial_features, compute_direction_labels
│   └── decoding_module.py              ← DecodingPipeline class (sklearn-compatible)
└── notebooks/
    ├── Phase3_FeatureExtraction_CV.ipynb    ← binned counts, PCA/ICA, scree plot, scatter
    └── Phase3_Decoding_Demo_CV.ipynb        ← regression + classification + cross-validation
```

---

## 9. Coding conventions (non-negotiable)

| Constraint | Rule |
|---|---|
| Dataset-agnostic | No hardcoded session IDs, dataset names, or column names |
| No S3 path hardcoding | Always use `Combined_Dataset_Utils` loaders |
| `shuffle=False` | ALL train/test splits and cross-validation |
| sklearn interface | `DecodingPipeline.fit/predict/score` must match sklearn API |
| No vx/vy from DANDI_000140 | Known duplication bug |
| Discrete decoding | Filter to `task_type in ('center_out', 'center_out_maze')` before `compute_direction_labels()` |
| Numpy-style docstrings | Every public function and method |
| `from __future__ import annotations` | Top of every `.py` file |

---

## 10. Critical gotchas

**spikes must be transposed for sklearn:**
```python
spikes = ds["spikes"].values    # (n_electrodes, n_time)
X = spikes.T                    # (n_time, n_electrodes) ← sklearn wants this
```
`compute_binned_counts()` handles this internally — do not double-transpose.

**trial_id == 0 is inter-trial (not -1):**  
Old code using `trial_id != -1` is wrong. Always use `trial_id != 0`.

**Wiener filter trims both X and y:**  
`_build_lag_matrix(X, n_lags)` returns `(n - n_lags, features*(n_lags+1))`. Trim y to `y[n_lags:]`.

**PCA n_components must be ≤ min(n_samples, n_features):**  
For small trial counts (DANDI_000688 ~180 trials), guard with `min(n_components, X.shape[0], X.shape[1])`.

**Zenodo_3854034 has no discrete trial structure:**  
`trial_phase` is always 2. `compute_direction_labels()` raises `ValueError` on this dataset by design.

**bin_phase sampling for scatter plots:**
```python
bin_phases = trial_phase_full[bin_size_ms // 2 :: bin_size_ms][:n_bins]
```
Sample the midpoint of each bin — more representative than first or last sample.

**combined_zarr.keys() returns empty on S3:**  
Always use `root.group_keys()` or `data_utils.filter_sessions()`.

---

## 11. Git conventions

- Branch naming: `feat/phase3-<description>-CV`
- Commit format (Conventional Commits):
  - `feat(decoding): add PCA feature extraction notebook`
  - `fix(decoding): guard PCA n_components against small trial counts`
  - `chore: update .gitignore`
- Never push directly to `main` — Juan Pablo reviews and merges PRs

---

## 12. Notebook style reference

- First cell: markdown with `# Phase 3 — [Name]`, description, structure table
- Import cell: ends with `print("✓ All imports successful")`
- S3 cell: ends with `print("✓ Connected to S3")`
- Each section: starts with markdown cell explaining the **WHY** before the code
- Visualization functions: follow `ax = None` pattern (can be called standalone or into existing axes)

---

## 13. Quality checklist (use before any PR)

- [ ] All files created: `decoding/__init__.py`, `decoding/feature_extraction.py`, `decoding/decoding_module.py`
- [ ] Notebooks created: `Phase3_FeatureExtraction_CV.ipynb`, `Phase3_Decoding_Demo_CV.ipynb`
- [ ] Gate 1: import check passes (no ImportError)
- [ ] Gate 2: synthetic data tests pass (all assertions green)
- [ ] `spikes.T` applied correctly inside `compute_binned_counts`
- [ ] `shuffle=False` in `evaluate_cv` — verified by inspection
- [ ] `compute_direction_labels` raises `ValueError` for non-center_out sessions
- [ ] `DecodingPipeline` stores fitted `scaler_` and `pca_` as instance attributes
- [ ] Wiener filter trims y by `n_lags` rows internally
- [ ] All public functions have numpy-style docstrings
- [ ] `from __future__ import annotations` at top of each `.py` file
- [ ] No hardcoded session IDs, dataset names, or S3 paths
