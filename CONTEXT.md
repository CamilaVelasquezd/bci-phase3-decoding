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
| Key packages | sklearn, numpy 1.26.4, scipy 1.15.3, seaborn 0.13.2 |
| Other packages | zarr, s3fs, xarray, dask, matplotlib, plotly |

The `bci-ds` environment is shared with `bci-decoding-dataset` — no new environment needed.  
If a new package is required, `pip install <pkg>` inside the activated `bci-ds` env.

---

## 3. The `bci_decoding_dataset` package — how to use it

The `bci-decoding-dataset` repo is installed as a Python package in the `bci-ds` environment. Import it directly — no need to copy any code from that repo.

### 3.1 New API (v0.8.0) — use these imports

```python
from bci_decoding_dataset import DatasetLoader
from bci_decoding_dataset.neuroviz import SessionPlots, DatasetPlots
```

**Do NOT use** `Combined_Dataset_Utils` — that is the old API. It still exists but is deprecated.

### 3.2 DatasetLoader — loading sessions from S3

`DatasetLoader` is the consumer-side class for reading sessions.

```python
import configparser
import os
from bci_decoding_dataset import DatasetLoader

# --- S3 connection (copy exactly) ---
credentials_path = os.path.expanduser("~/.aws/credentials")
config = configparser.ConfigParser()
config.read(credentials_path)
profile = os.environ.get("AWS_PROFILE", "cv-pc")

loader = DatasetLoader(
    aws_store=True,
    s3_bucket="solzbacher-lab-motor-decoding-ds",
    s3_key="datasets/Combined_Motor_Datasets",
    aws_access_key_id=config[profile]["aws_access_key_id"],
    aws_secret_access_key=config[profile]["aws_secret_access_key"],
)
```

### 3.3 DatasetLoader methods

| Method | Returns | Notes |
|---|---|---|
| `loader.filter_sessions(filter_by, filter_value)` | `list[str]` | Filter session IDs by attribute. Common: `"dataset_id"`, `"subject_id"`, `"task_type"` |
| `loader.get_processed_data_from_session(session_id)` | `xr.Dataset` | Lazy dataset — slice before `.values` |
| `loader.get_data_from_filter(filter_by, filter_value)` | `dict[str, xr.Dataset]` | Loads all matching sessions; skips failures |
| `loader.get_session_catalog()` | `list[dict]` | All sessions metadata; cached after first call |

```python
# List sessions for a dataset
dandi_70_sessions = loader.filter_sessions("dataset_id", "DANDI_00070")
session_id = dandi_70_sessions[0]

# Load a single session
ds = loader.get_processed_data_from_session(session_id)
print(ds.attrs["subject_id"], ds.attrs["task_type"])

# Load all Jenkins sessions at once
jenkins_data = loader.get_data_from_filter("subject_id", "Jenkins")
for sid, ds in jenkins_data.items():
    print(sid, ds["spikes"].shape)
```

### 3.4 SessionPlots — visualization (reuse, don't reimplement)

`SessionPlots` is constructed from a `DatasetLoader` + a session ID. All methods return `plotly.graph_objects.Figure`.

```python
from bci_decoding_dataset.neuroviz import SessionPlots

sp = SessionPlots(loader, session_id)
```

| Method | Purpose | Key params |
|---|---|---|
| `sp.pca(...)` | PCA trajectory on smoothed firing rates | `mode='continuous'\|'trial_aligned'`, `dim=2\|3`, `sigma_ms`, `t_start`, `t_end`, `n_start`, `n_end` |
| `sp.raster(...)` | Spike raster with optional trial shading | `t_start`, `t_end`, `n_start`, `n_end`, `show_trial_shading` |
| `sp.firing_rate(...)` | Smoothed firing rate curves | `sigma_ms`, `kernel='gaussian'\|'causal'` |
| `sp.heatmap(...)` | Firing rate heatmap (neurons × time) | `sort_by`, `zscore`, `bin_ms` |
| `sp.psth(...)` | Peri-stimulus time histogram | `event='trial_start'\|'reach_onset'\|'reach_offset'` |
| `sp.isi(...)` | Interspike interval histogram | Max 8 neurons per call |
| `sp.behavior(...)` | Firing rate + kinematic channels overlay | `channels=('position_x', 'position_y', ...)` |
| `sp.correlation(...)` | Neuron-neuron Pearson correlation matrix | `sort_by='cluster'\|'none'` |

**Critical distinction for Phase 3:**
- `sp.pca()` → use for **visualization only** (returns a Figure, not the PCA components)
- For the **decoding pipeline**, use `sklearn.decomposition.PCA` directly on the binned spike matrix

```python
# Visualization — use SessionPlots
sp.pca(t_start=0, t_end=10, n_start=0, n_end=49, mode="continuous").show()

# Decoding pipeline — use sklearn directly
from sklearn.decomposition import PCA
X = compute_binned_counts(ds, bin_size_ms=50)  # (n_bins, n_electrodes)
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)               # (n_bins, 10)
```

### 3.5 Warning: expensive SessionPlots calls

`sp.pca(mode="trial_aligned")` and `sp.psth()` load the **full session time axis** on every call (multi-GB on long recordings). Use `mode="continuous"` for exploratory work.

---

## 4. The four datasets

| Dataset | `dataset_id` | `task_type` | Electrodes | Sessions | Trials/session | Notes |
|---|---|---|---|---|---|---|
| DANDI:000070 | `DANDI_00070` | `center_out` | ~191 | 10 | ~3000 | **KNOWN BUG: vx == vy in ALL sessions** — do NOT use velocity as regression target. Use position displacement for direction labels instead. |
| DANDI:000688 | `DANDI_000688` | `center_out` | ~54 | 111 | ~180 | **✅ Preferred for continuous decoding** — vx ≠ vy confirmed. Shorter sessions = lower compute cost. |
| Zenodo:3854034 | `Zenodo_3854034` | `continuous_random` | ~96 | 47 | ~563 | No discrete trial structure. vx ≠ vy ✅ — velocity valid. |
| DANDI:000140 | `DANDI_000140` | `center_out_maze` | — | — | — | **KNOWN BUG: vx == vy in ALL sessions** — do NOT use velocity as regression target. |

**Velocity bug summary (confirmed by visual inspection):**
- `DANDI_00070` → vx == vy ❌ BUG
- `DANDI_000140` → vx == vy ❌ BUG
- `DANDI_000688` → vx ≠ vy ✅ OK — use this for continuous decoding
- `Zenodo_3854034` → vx ≠ vy ✅ OK — but no discrete trials

---

## 5. Data schema (V8)

Every session loaded via `loader.get_processed_data_from_session(session_id)` → `xr.Dataset`:

| Variable | Shape | Dtype | Notes |
|---|---|---|---|
| `spikes` | `(n_electrodes, n_time)` | uint8 | **Rows = electrodes, cols = time** — must transpose for sklearn |
| `position` | `(n_time, 2)` | float32 | Normalized x,y cursor position in [-1, 1] |
| `velocity` | `(n_time, 2)` | float32 | Normalized vx,vy in [-1, 1] — **do NOT use from DANDI_00070 or DANDI_000140** (vx==vy bug). DANDI_000688 and Zenodo_3854034 are safe. |
| `trial_id` | `(n_time,)` | int16 | 0=inter-trial, +N=successful trial N, -N=failed trial N |
| `trial_phase` | `(n_time,)` | int8 | 0=inter, 1=pre-reach, 2=reach, 3=post-reach |

Session attributes: `ds.attrs['task_type']`, `ds.attrs['dataset_id']`, `ds.attrs['subject_id']`, `ds.attrs['sampling_rate']` (always 1000.0 Hz).

---

## 6. Data access pattern

```python
# Load arrays into memory
spikes      = ds["spikes"].values         # (n_electrodes, n_time), uint8
position    = ds["position"].values       # (n_time, 2), float32
velocity    = ds["velocity"].values       # (n_time, 2), float32
trial_id    = ds["trial_id"].values       # (n_time,), int16
trial_phase = ds["trial_phase"].values    # (n_time,), int8

# Lazy access for large sessions (avoid loading full array into RAM)
window = ds["spikes"].isel(time=slice(0, 10000)).values  # only 10s

# Trial masks
active_mask          = trial_id != 0        # excludes inter-trial (V8: 0, NOT -1)
successful_mask      = trial_id > 0         # successful trials only
successful_trial_ids = np.unique(trial_id[trial_id > 0])
```

---

## 7. Phase 3 — Scientific plan

The decoding pipeline follows this sequence:

```
Spikes → Binning (50ms) → PCA → Continuous decoder → vx, vy
```

### 7.1 Current focus: Week 2 — Continuous decoding

**Dataset:** `DANDI_000688` — preferred by supervisor. Shorter sessions, lower compute cost, and velocity is confirmed correct (vx ≠ vy).

**Key decisions from supervisor (Juan Pablo Botero):**
- Use **active task bins only** — exclude inter-trial bins with `trial_phase > 0`.
- Use **PCA only** for dimensionality reduction — ICA and LDA are not used for the decoding pipeline.
- Target `y` = `ds["velocity"].values` — predict vx and vy at each time bin.
- The decoder is **bin-level**, not trial-level: one prediction per 50ms bin.

### 7.2 Dimensionality reduction pipeline

```python
# Step 1 — bin spikes, then keep active task bins
X_bins_all = compute_binned_counts(ds, bin_size_ms=50)  # (n_bins, n_electrodes)
bin_phases = compute_bin_phases(ds, bin_size_ms=50)
active_mask = bin_phases > 0
X_bins = X_bins_all[active_mask]

velocity = ds["velocity"].values
n_bins = X_bins_all.shape[0]
y_all = velocity[: n_bins * 50].reshape(n_bins, 50, 2).mean(axis=1)
y = y_all[active_mask]                              # (n_active_bins, 2) — vx and vy

# Step 2 — reduce with PCA
reducer = DimReducer(method='pca', n_components=10)
X_reduced = reducer.fit_transform(X_bins)           # (n_bins, 10)
```

**Why exclude inter-trial bins:** Continuous decoding should focus on neural states during active task performance. Keep pre-reach, reach, and post-reach bins (`trial_phase > 0`) and drop inter-trial bins (`trial_phase == 0`).

### 7.3 Continuous decoding models (this week)

| Model | Notes |
|---|---|
| **Ridge regression** | Regularized linear regression — first benchmark |
| **Wiener filter** | Ridge with temporal lags — classical BCI baseline |
| **Kalman filter** | Probabilistic state-space model — implement after linear baselines |

Metric: R² score per velocity dimension (vx and vy separately). `shuffle=False` always.

```python
# Target alignment with Wiener filter (n_lags > 0)
# _build_lag_matrix returns (n_bins - n_lags, n_features * (n_lags+1))
# y must be trimmed: y = y[n_lags:]
```

### 7.4 Discrete decoding (NEXT WEEK — do not implement yet)

Method TBD — supervisor will share a paper. Will involve quantizing velocity into ranges. Dataset and approach to be confirmed. Do not use direction labels or `compute_direction_labels_from_position` for this.

---

## 8. Repo structure (current)

```
bci-phase3-decoding/
├── CONTEXT.md
├── README.md
├── .claude/
│   ├── commands/
│   ├── agents/
│   └── skills/
├── decoding/
│   ├── __init__.py
│   ├── data_loading.py      ← S3 connection, session loading
│   ├── dim_reduction.py     ← compute_binned_counts, compute_bin_phases, DimReducer
│   ├── discrete_utils.py    ← trial averaging, direction labels (NOT used for continuous decoders)
│   └── regression.py        ← continuous decoders: Ridge, Wiener, Kalman
└── notebooks/
    ├── Phase3_DimReduction_CV.ipynb      ← Week 1 exploration (PCA/ICA/LDA on X_trials — kept for reference)
    ├── Phase3_DimReduction_v2_CV.ipynb   ← Week 1 v2 with DimReducer
    └── Phase3_ContinuousDecoding_CV.ipynb ← Week 2: Ridge, Wiener, Kalman on DANDI_000688
```

---

## 9. Coding conventions (non-negotiable)

| Constraint | Rule |
|---|---|
| Use new API | `from bci_decoding_dataset import DatasetLoader` — not `Combined_Dataset_Utils` |
| Dataset-agnostic | No hardcoded session IDs, dataset names, or column names |
| No S3 path hardcoding | Always use `DatasetLoader` with credentials from `~/.aws/credentials` |
| `shuffle=False` | ALL train/test splits and cross-validation |
| sklearn interface | `DecodingPipeline.fit/predict/score` must match sklearn API |
| No vx/vy from DANDI_00070 or DANDI_000140 | vx == vy duplication bug confirmed in both datasets — use DANDI_000688 for continuous decoding |
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
For DANDI_000688 (~180 trials), guard with `min(n_components, X.shape[0], X.shape[1])`.

**vx == vy bug affects DANDI_00070 AND DANDI_000140:**
Confirmed by visual inspection — both datasets have identical vx and vy columns. Use `DANDI_000688` for all continuous decoding work. Zenodo_3854034 is also velocity-safe but has no discrete trial structure.

**Filter out inter-trial bins for continuous decoding:**
Use active task bins only. The full pipeline bins all spikes first, computes `bin_phases`, then keeps `bin_phases > 0`.

**X_bins vs X_trials:**
- `X_bins` — active bins, no averaging, shape `(n_active_bins, n_electrodes)` — use for continuous decoding
- `X_trials` — reach-phase average per trial, shape `(n_trials, n_electrodes)` — only for discrete decoding exploration (in `discrete_utils.py`)

**bin_phase sampling for scatter plots:**
```python
bin_phases = trial_phase_full[bin_size_ms // 2 :: bin_size_ms][:n_bins]
```
Sample the midpoint of each bin.

**SessionPlots.pca() vs sklearn PCA:**  
`sp.pca()` returns a Figure — it does not expose the components.  
For the decoding pipeline, always use `sklearn.decomposition.PCA` directly.

**Lazy loading — slice before `.values`:**  
`ds["spikes"].values` loads the entire session into RAM.  
Use `ds["spikes"].isel(time=slice(0, 10000)).values` for exploration.

---

## 11. Git conventions

- Commit format (Conventional Commits):
  - `feat(notebooks): add PCA dimensionality reduction notebook`
  - `feat(decoding): add compute_binned_counts function`
  - `fix(decoding): guard PCA n_components against small trial counts`
  - `chore: update .gitignore`
- Work directly on `main` (solo repo)

---

## 12. Notebook style reference

- First cell: markdown with `# Phase 3 — [Name]`, description, structure table
- Import cell: ends with `print("✓ All imports successful")`
- S3 cell: ends with `print("✓ Connected to S3")`
- Each section: starts with markdown cell explaining the **WHY** before the code
- Visualization: use `SessionPlots` methods for spike plots; matplotlib/seaborn for custom decoding plots

---

## 13. Quality checklist

- [ ] Using `DatasetLoader` (not `Combined_Dataset_Utils`)
- [ ] S3 credentials loaded from `~/.aws/credentials` with profile `cv-pc`
- [ ] `spikes.T` applied correctly (or handled inside `compute_binned_counts`)
- [ ] `trial_id != 0` used (not `!= -1`)
- [ ] `shuffle=False` in all train/test splits
- [ ] PCA n_components guarded against small datasets
- [ ] Using `DANDI_000688` for continuous decoding (not DANDI_00070 or Zenodo)
- [ ] Inter-trial bins removed with `bin_phases > 0`
- [ ] `y` is binned/mean-pooled and masked with the same `active_mask` as `X_bins`
- [ ] Wiener filter: y trimmed by n_lags rows to match lag matrix
- [ ] No hardcoded session IDs, dataset names, or S3 paths
- [ ] All public functions have numpy-style docstrings
- [ ] `from __future__ import annotations` at top of each `.py` file
