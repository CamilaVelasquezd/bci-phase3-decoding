# PRP — FR001: Phase 3 Classical Decoding Module
**Project:** Phase 3 - Classical decoding & Machine learning 

**Author:** Camila Velásquez Díaz (practicum intern, Solzbacher Lab, University of Utah)

**Supervisor:** Juan Pablo Botero 

**Confidence score:** 8/10 — one-pass implementation is achievable with this context

---

## 1. Goal

Build the `decoding/` package at the repo root. It must contain:

| File | Purpose |
|---|---|
| `decoding/__init__.py` | Exports `DecodingPipeline` + feature functions |
| `decoding/feature_extraction.py` | `compute_binned_counts()`, `smooth_firing_rates()`, `compute_trial_features()`, `compute_direction_labels()` |
| `decoding/decoding_module.py` | `DecodingPipeline` class (sklearn-compatible) |
| `decoding/notebooks/Phase3_FeatureExtraction_CV.ipynb` | Week 7 deliverable — binned counts, PCA, scree plot, PC1/PC2 scatter |
| `decoding/notebooks/Phase3_Decoding_Demo_CV.ipynb` | Week 10 scaffold — continuous + discrete decoding + cross-validation |

This folder sits at the repo root, same level as `neuroviz/` and `Tutorials/`.

---

## 2. Context You Must Know

### 2.1 Environment
- Conda env: `bci-ds` (Python 3.10, Windows)
- AWS profile: `cv-pc` stored in `~/.aws/credentials`
- S3 bucket: `solzbacher-lab-motor-decoding-ds`
- S3 path: `datasets/Combined_Motor_Datasets`
- sklearn v1.7.2, numpy 1.26.4, scipy 1.15.3, seaborn 0.13.2 — all installed

### 2.2 The xr.Dataset schema (V8)

Every session's `ds = data_utils.get_processed_data_from_session(session_id)` gives:

| Variable | Shape | Dtype | Notes |
|---|---|---|---|
| `spikes` | `(n_electrodes, n_time)` | uint8 | **Rows = electrodes, cols = time** — transpose for sklearn |
| `position` | `(n_time, 2)` | float64 | Normalized x,y cursor position |
| `velocity` | `(n_time, 2)` | float64 | Normalized x,y cursor velocity |
| `trial_id` | `(n_time,)` | int16 | 0=inter, +N=success trial N, -N=fail trial N |
| `trial_phase` | `(n_time,)` | int8 | 0=inter, 1=pre-reach, 2=reach, 3=post-reach |

Session attributes: `ds.attrs['task_type']`, `ds.attrs['dataset_id']`, `ds.attrs['sampling_rate']` (always 1000.0).

### 2.3 The four datasets and their quirks

| Dataset | `task_type` | Notes |
|---|---|---|
| DANDI_00070 | `'center_out'` | ~191 electrodes, ~3000 trials; best for discrete decoding |
| DANDI_000688 | `'center_out'` | ~54 electrodes, ~180 trials per session |
| Zenodo_3854034 | `'continuous_random'` | No discrete trials — `trial_phase` is always 2; **skip for discrete decoding** |
| DANDI_000140 | `'center_out_maze'` | **KNOWN BUG: vx and vy are duplicated (same values). Do NOT use `velocity` from this dataset for regression targets.** Use position or skip it. |

### 2.4 S3 connection pattern (EXACT — copy from existing notebooks)

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

# Direct S3 (needed for listing sessions — combined_zarr.keys() returns empty)
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

# Filter sessions by dataset
dandi_70 = data_utils.filter_sessions(filter_by="dataset_id", filter_value="DANDI_00070")
session_id = dandi_70[0]
ds = data_utils.get_processed_data_from_session(session_id)
```

**WARNING**: `data_utils.combined_zarr.keys()` returns empty on S3. Always use `root.group_keys()` or `data_utils.filter_sessions()` to list sessions.

### 2.5 Data access pattern

```python
# Load arrays into memory (force eager evaluation from dask)
spikes = ds["spikes"].values         # (n_electrodes, n_time), uint8
position = ds["position"].values     # (n_time, 2), float64
velocity = ds["velocity"].values     # (n_time, 2), float64
trial_id = ds["trial_id"].values     # (n_time,), int16
trial_phase = ds["trial_phase"].values  # (n_time,), int8

# Trial handling
active_mask = trial_id != 0           # excludes inter-trial periods
successful_mask = trial_id > 0        # successful trials only
successful_trial_ids = np.unique(trial_id[trial_id > 0])  # sorted trial IDs
```

---

## 3. Files to Create — Implementation Blueprint

### 3.1 `decoding/feature_extraction.py`

**Purpose**: Pure numpy functions that convert `xr.Dataset` → feature matrices suitable for sklearn.

```python
"""
feature_extraction.py
─────────────────────
Converts Zarr V8 sessions (xr.Dataset) into sklearn-ready feature matrices.

Functions
---------
compute_binned_counts  : spike train → binned count matrix (time_bins × neurons)
smooth_firing_rates    : apply Gaussian smoothing to binned counts
compute_trial_features : average binned counts per trial (trials × neurons)
compute_direction_labels : derive 8-direction labels from cursor trajectory
"""

from __future__ import annotations
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d


def compute_binned_counts(ds: xr.Dataset, bin_size_ms: int = 50) -> np.ndarray:
    """
    Bin spike trains into spike count vectors for ML feature input.

    Parameters
    ----------
    ds : xr.Dataset
        Processed session dataset with 'spikes' variable of shape
        (n_electrodes, n_time) at 1 kHz.
    bin_size_ms : int
        Bin size in milliseconds. Must divide n_time cleanly; excess
        timepoints are dropped. Default 50 ms.

    Returns
    -------
    X : np.ndarray, shape (n_bins, n_electrodes)
        Spike count matrix. Each row is one time bin; each column is one
        electrode. Suitable for direct use as sklearn feature matrix.

    Examples
    --------
    >>> X = compute_binned_counts(ds, bin_size_ms=50)
    >>> X.shape  # (n_time // 50, n_electrodes)
    """
    spikes = ds["spikes"].values              # (n_electrodes, n_time)
    n_electrodes, n_time = spikes.shape
    n_bins = n_time // bin_size_ms
    spikes_trimmed = spikes[:, : n_bins * bin_size_ms]
    # reshape → (n_electrodes, n_bins, bin_size_ms), sum over last axis
    X = spikes_trimmed.reshape(n_electrodes, n_bins, bin_size_ms).sum(axis=2)
    return X.T.astype(np.float32)             # (n_bins, n_electrodes)


def smooth_firing_rates(X: np.ndarray, sigma_bins: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to a binned spike count matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_bins, n_electrodes)
        Binned spike count matrix from compute_binned_counts().
    sigma_bins : float
        Gaussian kernel SD in units of bins. Default 2.0 bins.
        For 50 ms bins, sigma_bins=2 → sigma_ms=100 ms.

    Returns
    -------
    X_smooth : np.ndarray, shape (n_bins, n_electrodes)
        Smoothed firing rate estimates (counts/bin, smoothed).
    """
    return gaussian_filter1d(X.astype(np.float32), sigma=sigma_bins, axis=0)


def compute_trial_features(
    ds: xr.Dataset,
    phase: int = 2,
    successful_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-trial mean firing rates for discrete decoding.

    Averages spike counts across the specified phase window within each trial.
    Only electrodes with at least one spike are returned.

    Parameters
    ----------
    ds : xr.Dataset
        Processed session dataset.
    phase : int
        trial_phase value to average over. Default 2 (reach).
    successful_only : bool
        If True (default), include only trials where trial_id > 0.

    Returns
    -------
    X_trials : np.ndarray, shape (n_trials, n_electrodes)
        Mean firing rate per electrode per trial.
    trial_ids : np.ndarray, shape (n_trials,)
        Signed trial IDs corresponding to each row of X_trials.

    Notes
    -----
    Returns empty arrays if no valid trials are found.
    """
    spikes = ds["spikes"].values           # (n_electrodes, n_time)
    trial_id = ds["trial_id"].values       # (n_time,)
    trial_phase = ds["trial_phase"].values # (n_time,)

    if successful_only:
        valid_ids = np.unique(trial_id[trial_id > 0])
    else:
        valid_ids = np.unique(np.abs(trial_id[trial_id != 0]))

    rows, ids = [], []
    for tid in valid_ids:
        if successful_only:
            mask = (trial_id == tid) & (trial_phase == phase)
        else:
            mask = (np.abs(trial_id) == tid) & (trial_phase == phase)
        if mask.sum() == 0:
            continue
        rows.append(spikes[:, mask].mean(axis=1))  # mean over time, shape (n_electrodes,)
        ids.append(tid if successful_only else int(tid))

    if not rows:
        return np.empty((0, spikes.shape[0])), np.empty(0, dtype=int)

    return np.stack(rows, axis=0).astype(np.float32), np.array(ids)


def compute_direction_labels(
    ds: xr.Dataset,
    n_directions: int = 8,
    successful_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive discrete movement direction labels from cursor trajectory.

    Computes the dominant movement direction for each trial by measuring
    the angle from the cursor position at reach onset to the position at
    reach offset, then quantizes to n_directions equally spaced bins.

    Parameters
    ----------
    ds : xr.Dataset
        Processed session dataset. Must have 'position', 'trial_id',
        'trial_phase' variables. task_type must be 'center_out' or
        'center_out_maze'.
    n_directions : int
        Number of direction classes. Default 8 (0° to 315° in 45° steps).
    successful_only : bool
        If True (default), include only successful trials.

    Returns
    -------
    labels : np.ndarray, shape (n_trials,), dtype int
        Direction label in [0, n_directions - 1] for each trial.
    trial_ids : np.ndarray, shape (n_trials,)
        Signed trial IDs corresponding to each label.

    Raises
    ------
    ValueError
        If task_type is not 'center_out' or 'center_out_maze'.

    Notes
    -----
    Direction 0 corresponds to 0° (rightward), increasing counter-clockwise.
    Labels are derived from np.arctan2(dy, dx) quantized to n_directions bins.
    """
    task_type = ds.attrs.get("task_type", "")
    if task_type not in ("center_out", "center_out_maze"):
        raise ValueError(
            f"compute_direction_labels requires center_out or center_out_maze task, "
            f"got '{task_type}'. Skip discrete decoding for '{task_type}' sessions."
        )

    position = ds["position"].values       # (n_time, 2)
    trial_id = ds["trial_id"].values       # (n_time,)
    trial_phase = ds["trial_phase"].values # (n_time,)

    if successful_only:
        valid_ids = np.unique(trial_id[trial_id > 0])
    else:
        valid_ids = np.unique(np.abs(trial_id[trial_id != 0]))

    labels, ids = [], []
    step = 2 * np.pi / n_directions

    for tid in valid_ids:
        if successful_only:
            reach_mask = (trial_id == tid) & (trial_phase == 2)
        else:
            reach_mask = (np.abs(trial_id) == tid) & (trial_phase == 2)

        if reach_mask.sum() < 2:
            continue

        reach_pos = position[reach_mask]      # (n_reach_time, 2)
        dx = reach_pos[-1, 0] - reach_pos[0, 0]
        dy = reach_pos[-1, 1] - reach_pos[0, 1]
        angle = np.arctan2(dy, dx)            # in (-π, π]
        direction = int(round(angle / step)) % n_directions
        labels.append(direction)
        ids.append(tid if successful_only else int(tid))

    return np.array(labels, dtype=int), np.array(ids)
```

### 3.2 `decoding/decoding_module.py`

**Purpose**: `DecodingPipeline` class — sklearn-compatible API wrapping StandardScaler → PCA → model.

```python
"""
decoding_module.py
──────────────────
DecodingPipeline: unified interface for neural decoding.
Mirrors sklearn's fit / predict / score API.

Supported models
----------------
continuous  : 'linear' (LinearRegression), 'ridge' (Ridge), 'wiener' (Ridge + temporal lags)
discrete    : 'lda' (LinearDiscriminantAnalysis), 'svm' (SVC, linear kernel)

Pipeline
--------
StandardScaler → PCA(n_components) → model

For Wiener filter (n_lags > 0), lagged features are built before scaling:
  X_lagged[t] = [X[t], X[t-1], ..., X[t-n_lags]]
  shape: (n_samples - n_lags, n_features * (n_lags + 1))
  The corresponding y rows are trimmed to match.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


_CONTINUOUS_MODELS = {"linear", "ridge", "wiener"}
_DISCRETE_MODELS   = {"lda", "svm"}
_ALL_MODELS        = _CONTINUOUS_MODELS | _DISCRETE_MODELS


def _build_lag_matrix(X: np.ndarray, n_lags: int) -> np.ndarray:
    """Stack current and past n_lags feature vectors column-wise.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    n_lags : int

    Returns
    -------
    X_lagged : np.ndarray, shape (n_samples - n_lags, n_features * (n_lags + 1))
    """
    if n_lags == 0:
        return X
    n, f = X.shape
    parts = [X[n_lags - lag : n - lag if lag > 0 else n] for lag in range(n_lags + 1)]
    return np.hstack(parts)


class DecodingPipeline:
    """
    Unified interface for neural decoding pipelines.
    Mirrors sklearn's fit / predict / score API.

    Parameters
    ----------
    model_type : str
        One of: 'linear', 'ridge', 'wiener', 'lda', 'svm'.
    task_type : str
        'continuous' (regression) or 'discrete' (classification).
    n_components_pca : int
        Number of PCA components. Default 10.
    n_lags : int
        Temporal lag bins for Wiener filter (model_type='wiener'). Default 0.
    ridge_alpha : float
        Regularization strength for Ridge. Default 1.0.

    Examples
    --------
    >>> pipeline = DecodingPipeline(model_type='ridge', task_type='continuous')
    >>> pipeline.fit(X_train, y_train)
    >>> r2 = pipeline.score(X_test, y_test)
    >>> scores = pipeline.evaluate_cv(X, y, n_splits=5)
    """

    def __init__(
        self,
        model_type: str = "ridge",
        task_type: str = "continuous",
        n_components_pca: int = 10,
        n_lags: int = 0,
        ridge_alpha: float = 1.0,
    ) -> None:
        if model_type not in _ALL_MODELS:
            raise ValueError(f"model_type must be one of {sorted(_ALL_MODELS)}")
        if task_type not in ("continuous", "discrete"):
            raise ValueError("task_type must be 'continuous' or 'discrete'")

        self.model_type        = model_type
        self.task_type         = task_type
        self.n_components_pca  = n_components_pca
        self.n_lags            = n_lags
        self.ridge_alpha       = ridge_alpha

        # Fitted state (None until fit() is called)
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None               = None
        self.model_: object | None          = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_model(self):
        if self.model_type == "linear":
            return LinearRegression()
        if self.model_type in ("ridge", "wiener"):
            return Ridge(alpha=self.ridge_alpha)
        if self.model_type == "lda":
            return LinearDiscriminantAnalysis()
        if self.model_type == "svm":
            return SVC(kernel="linear")
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def _prepare_X(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Apply lag → scale → PCA. fit=True also fits scaler and PCA."""
        if self.model_type == "wiener" and self.n_lags > 0:
            X = _build_lag_matrix(X, self.n_lags)

        n_comp = min(self.n_components_pca, X.shape[0], X.shape[1])

        if fit:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            self.pca_ = PCA(n_components=n_comp)
            X_out = self.pca_.fit_transform(X_scaled)
        else:
            X_scaled = self.scaler_.transform(X)
            X_out = self.pca_.transform(X_scaled)

        return X_out

    def _trim_y(self, y: np.ndarray, n_lags: int) -> np.ndarray:
        """Trim the first n_lags rows of y to match the lagged feature matrix."""
        if self.model_type == "wiener" and n_lags > 0:
            return y[n_lags:]
        return y

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecodingPipeline":
        """
        Fit the pipeline: build lags (if Wiener) → StandardScaler → PCA → model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix (time_bins × electrodes or trials × electrodes).
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            Regression targets or class labels.

        Returns
        -------
        self
        """
        X_proc = self._prepare_X(X, fit=True)
        y_proc = self._trim_y(y, self.n_lags)
        self.model_ = self._make_model()
        self.model_.fit(X_proc, y_proc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted pipeline to new data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray
        """
        if self.scaler_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_proc = self._prepare_X(X, fit=False)
        return self.model_.predict(X_proc)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² (continuous) or accuracy (discrete) on held-out data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        score : float
        """
        y_pred = self.predict(X)
        y_proc = self._trim_y(y, self.n_lags)
        if self.task_type == "continuous":
            return float(r2_score(y_proc, y_pred))
        return float(accuracy_score(y_proc, y_pred))

    def evaluate_cv(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5
    ) -> np.ndarray:
        """
        Cross-validate with KFold(shuffle=False) to preserve temporal structure.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
        n_splits : int
            Number of CV folds. Default 5.

        Returns
        -------
        scores : np.ndarray, shape (n_splits,)
            Per-fold R² (continuous) or accuracy (discrete).

        Notes
        -----
        shuffle=False is mandatory for neural time-series data because
        consecutive trials are temporally correlated. Shuffling leaks
        block-level drift into the test set, inflating performance estimates.
        """
        if self.task_type == "continuous":
            cv = KFold(n_splits=n_splits, shuffle=False)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=False)

        scores = []
        for train_idx, test_idx in cv.split(X, y if self.task_type == "discrete" else None):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            self.fit(X_tr, y_tr)
            scores.append(self.score(X_te, y_te))

        return np.array(scores)
```

### 3.3 `decoding/__init__.py`

```python
"""decoding — classical neural decoding utilities for the Solzbacher Lab BCI pipeline."""

from decoding.decoding_module import DecodingPipeline
from decoding.feature_extraction import (
    compute_binned_counts,
    compute_direction_labels,
    compute_trial_features,
    smooth_firing_rates,
)

__all__ = [
    "DecodingPipeline",
    "compute_binned_counts",
    "smooth_firing_rates",
    "compute_trial_features",
    "compute_direction_labels",
]
```

### 3.4 Notebook: `Phase3_FeatureExtraction_CV.ipynb`

Follow the **exact same structure** as `demos/Phase2_SpikeAnalysis_Visualization_CV_1.ipynb`. Mirror:
- Header markdown cell with title, scope, structure table
- Setup section reusing the exact S3 connection pattern above
- Output cells confirming each step with `print("✓ ...")`

**Cell structure (create as a proper Jupyter .ipynb):**

```
Cell 0 (markdown): Title — Phase 3 | Feature Extraction | Author | Date
Cell 1 (markdown): ## 0. Setup & Data Loading
Cell 2 (code):     S3 connection (exact pattern from §2.4)
Cell 3 (markdown): ## 1. Binned Spike Count Matrix
Cell 4 (code):     from decoding.feature_extraction import compute_binned_counts
                   X = compute_binned_counts(ds, bin_size_ms=50)
                   print(f"Feature matrix shape: {X.shape}  (n_bins × n_electrodes)")
Cell 5 (markdown): ## 2. PCA — Dimensionality Reduction
Cell 6 (code):     from sklearn.preprocessing import StandardScaler
                   from sklearn.decomposition import PCA
                   scaler = StandardScaler()
                   X_scaled = scaler.fit_transform(X)
                   pca = PCA(n_components=10)
                   X_pca = pca.fit_transform(X_scaled)
Cell 7 (code):     # Scree plot — explained variance per component
                   fig, ax = plt.subplots(figsize=(8, 4))
                   ax.bar(range(1, 11), pca.explained_variance_ratio_ * 100, ...)
Cell 8 (code):     # PC1 vs PC2 scatter colored by trial_phase
                   # get trial_phase at bin level (one phase value per bin)
                   trial_phase_full = ds["trial_phase"].values
                   n_bins = X.shape[0]; bin_size_ms = 50
                   # sample middle of each bin
                   bin_phases = trial_phase_full[bin_size_ms // 2 :: bin_size_ms][:n_bins]
                   phase_colors = {0: "gray", 1: "royalblue", 2: "crimson", 3: "seagreen"}
Cell 9 (markdown): ## 3. FastICA comparison (optional)
Cell 10 (code):    from sklearn.decomposition import FastICA ...
```

**KEY implementation detail for PC scatter:**
The `trial_phase` is at 1ms resolution. To get one label per bin, sample the middle of each bin:
```python
trial_phase_full = ds["trial_phase"].values         # (n_time,)
bin_phases = trial_phase_full[bin_size_ms // 2 :: bin_size_ms][:n_bins]
```

### 3.5 Notebook: `Phase3_Decoding_Demo_CV.ipynb`

This is the Week 10 final demo. For Week 7/8/9 it should be scaffolded with clear section stubs.

**Full outline** (create placeholder cells for Week 8/9/10):

```
Section 0: Setup & Data Loading  (functional — same S3 pattern)
Section 1: Feature Extraction    (functional — from Phase3_FeatureExtraction_CV)
Section 2: Continuous Decoding   (Week 8 — Ridge + Wiener filter)
  2.1  Target: velocity (vx, vy) from DANDI_00070 or DANDI_000688 only
  2.2  Train/test split (shuffle=False, 80/20)
  2.3  LinearRegression baseline
  2.4  Ridge (alpha sweep: 0.01, 0.1, 1.0, 10.0)
  2.5  Wiener filter (n_lags sweep: 0, 3, 5, 8, 10)
  2.6  Visualization: y_pred vs y_test over time
  2.7  R² vs lag-length plot
Section 3: Discrete Decoding     (Week 9 — LDA + SVM)
  3.1  Filter to center_out sessions only (skip Zenodo_3854034)
  3.2  Derive direction labels with compute_direction_labels()
  3.3  DummyClassifier baseline (chance = 1/8 = 12.5%)
  3.4  LDA — accuracy, confusion matrix, LDA embedding scatter
  3.5  SVM (kernel='linear')
  3.6  Comparison table
Section 4: Cross-Validation      (Week 10)
  4.1  KFold(5, shuffle=False) for continuous
  4.2  StratifiedKFold(5, shuffle=False) for discrete
  4.3  Results table: mean ± std
```

---

## 4. Implementation Tasks (Ordered)

1. Create directory `decoding/` and `decoding/notebooks/`
2. Write `decoding/feature_extraction.py` (exact code above, verified against data shapes)
3. Write `decoding/decoding_module.py` (exact code above)
4. Write `decoding/__init__.py`
5. Write `decoding/notebooks/Phase3_FeatureExtraction_CV.ipynb` as a proper Jupyter notebook JSON
6. Write `decoding/notebooks/Phase3_Decoding_Demo_CV.ipynb` (scaffold — Week 8/9/10 cells can be stubs with markdown placeholders)
7. **Self-validate** using the synthetic data tests below (no S3 required)

---

## 5. Validation Gates

### Gate 1 — Import check (no S3 needed)
```bash
cd c:/Git_practica/bci-decoding-dataset
python -c "
import sys; sys.path.insert(0, '.')
from decoding import DecodingPipeline, compute_binned_counts
from decoding.feature_extraction import compute_trial_features, compute_direction_labels, smooth_firing_rates
print('✓ All imports successful')
"
```

### Gate 2 — Synthetic data unit test (no S3 needed)
```python
# Run this as: python -c "exec(open('.agents/PRPs/feature-request/FR001-phase3-decoding/test_synthetic.py').read())"
# OR paste directly in Python REPL

import sys; sys.path.insert(0, ".")
import numpy as np
import xarray as xr

from decoding.feature_extraction import compute_binned_counts, smooth_firing_rates, compute_trial_features, compute_direction_labels
from decoding.decoding_module import DecodingPipeline, _build_lag_matrix

# --- Build a minimal mock xr.Dataset ---
np.random.seed(42)
n_electrodes, n_time = 50, 10_000
SR = 1000  # Hz

spikes_arr = np.random.binomial(1, 0.02, (n_electrodes, n_time)).astype(np.uint8)
position_arr = np.random.uniform(-1, 1, (n_time, 2)).astype(np.float64)
velocity_arr = np.random.uniform(-0.1, 0.1, (n_time, 2)).astype(np.float64)
trial_id_arr = np.zeros(n_time, dtype=np.int16)
trial_phase_arr = np.zeros(n_time, dtype=np.int8)

# Simulate 10 successful center-out trials, each 800ms with 3 phases
for i in range(10):
    t0 = i * 900 + 50
    # pre-reach 200ms (phase 1)
    trial_id_arr[t0:t0+200] = i + 1
    trial_phase_arr[t0:t0+200] = 1
    # reach 400ms (phase 2) — cursor moves in fixed direction
    trial_id_arr[t0+200:t0+600] = i + 1
    trial_phase_arr[t0+200:t0+600] = 2
    direction_rad = i * (2 * np.pi / 8)
    for j in range(400):
        position_arr[t0+200+j, 0] = j/400 * np.cos(direction_rad)
        position_arr[t0+200+j, 1] = j/400 * np.sin(direction_rad)
    # post-reach 200ms (phase 3)
    trial_id_arr[t0+600:t0+800] = i + 1
    trial_phase_arr[t0+600:t0+800] = 3

ds_mock = xr.Dataset(
    {
        "spikes":      (["electrode", "time"], spikes_arr),
        "position":    (["time", "coord"], position_arr),
        "velocity":    (["time", "coord"], velocity_arr),
        "trial_id":    (["time"], trial_id_arr),
        "trial_phase": (["time"], trial_phase_arr),
    },
    coords={"electrode": np.arange(n_electrodes), "coord": ["x", "y"],
            "time": np.arange(n_time) / SR},
    attrs={"task_type": "center_out", "dataset_id": "MOCK", "sampling_rate": 1000.0},
)

# --- Test compute_binned_counts ---
X = compute_binned_counts(ds_mock, bin_size_ms=50)
expected_bins = n_time // 50
assert X.shape == (expected_bins, n_electrodes), f"Expected ({expected_bins}, {n_electrodes}), got {X.shape}"
assert X.dtype == np.float32
print(f"✓ compute_binned_counts: shape {X.shape}")

# --- Test smooth_firing_rates ---
X_smooth = smooth_firing_rates(X, sigma_bins=2.0)
assert X_smooth.shape == X.shape
print(f"✓ smooth_firing_rates: shape {X_smooth.shape}")

# --- Test compute_trial_features ---
X_trials, trial_ids = compute_trial_features(ds_mock, phase=2, successful_only=True)
assert X_trials.shape[0] == 10, f"Expected 10 trials, got {X_trials.shape[0]}"
assert X_trials.shape[1] == n_electrodes
print(f"✓ compute_trial_features: shape {X_trials.shape}, trial_ids {trial_ids}")

# --- Test compute_direction_labels ---
labels, ids = compute_direction_labels(ds_mock, n_directions=8)
assert len(labels) == 10
assert all(0 <= l < 8 for l in labels), f"Labels out of range: {labels}"
print(f"✓ compute_direction_labels: {labels}")

# --- Test _build_lag_matrix ---
X_small = np.arange(30).reshape(10, 3).astype(float)
X_lagged = _build_lag_matrix(X_small, n_lags=2)
assert X_lagged.shape == (8, 9), f"Expected (8, 9), got {X_lagged.shape}"
print(f"✓ _build_lag_matrix: shape {X_lagged.shape}")

# --- Test DecodingPipeline: ridge continuous ---
pipeline = DecodingPipeline(model_type="ridge", task_type="continuous", n_components_pca=5)
y_cont = np.random.randn(expected_bins, 2).astype(np.float32)
split = int(expected_bins * 0.8)
pipeline.fit(X[:split], y_cont[:split])
y_pred = pipeline.predict(X[split:])
assert y_pred.shape == (expected_bins - split, 2)
score = pipeline.score(X[split:], y_cont[split:])
print(f"✓ DecodingPipeline (ridge/continuous): R² = {score:.4f}")

# --- Test DecodingPipeline: lda discrete ---
pipeline_lda = DecodingPipeline(model_type="lda", task_type="discrete", n_components_pca=5)
n_tr = X_trials.shape[0]
split_tr = n_tr // 2
pipeline_lda.fit(X_trials[:split_tr], labels[:split_tr])
acc = pipeline_lda.score(X_trials[split_tr:], labels[split_tr:])
print(f"✓ DecodingPipeline (lda/discrete): accuracy = {acc:.4f}")

# --- Test evaluate_cv ---
cv_scores = pipeline.evaluate_cv(X, y_cont, n_splits=3)
assert cv_scores.shape == (3,)
print(f"✓ evaluate_cv (ridge): fold R² = {cv_scores.round(4)}")

# --- Test wiener filter pipeline ---
pipeline_wiener = DecodingPipeline(model_type="wiener", task_type="continuous", n_components_pca=5, n_lags=3)
pipeline_wiener.fit(X[:split], y_cont[:split])
y_wiener_pred = pipeline_wiener.predict(X[split:])
print(f"✓ DecodingPipeline (wiener, n_lags=3): pred shape {y_wiener_pred.shape}")

# --- ValueError on continuous_random task ---
ds_zenodo = ds_mock.assign_attrs(task_type="continuous_random")
try:
    compute_direction_labels(ds_zenodo)
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"✓ ValueError on non-center_out task: {e}")

print()
print("═══════════════════════════════════════════════════════")
print("✅  All synthetic data validation gates passed!")
print("═══════════════════════════════════════════════════════")
```

### Gate 3 — Save the test file and run it
```bash
cd c:/Git_practica/bci-decoding-dataset
python -c "
import sys; sys.path.insert(0, '.')
# paste or exec the Gate 2 test script
"
```

If all assertions pass, the module is functionally correct against the V8 schema.

---

## 6. Coding Conventions to Follow

These are **non-negotiable constraints** from the lab:

| Constraint | Rule |
|---|---|
| Dataset-agnostic | No hardcoded session IDs, dataset names, or column names |
| No S3 path hardcoding | Always use `Combined_Dataset_Utils` loaders |
| `shuffle=False` | ALL train/test splits and cross-validation |
| sklearn interface | `DecodingPipeline.fit/predict/score` must match sklearn's API |
| No vx/vy from DANDI_000140 | Known duplication bug — use other datasets for velocity targets |
| Discrete decoding | Filter to `task_type in ('center_out', 'center_out_maze')` before calling `compute_direction_labels()` |
| Numpy-style docstrings | Every public function and method |
| `from __future__ import annotations` | At the top of every `.py` file |

---

## 7. Gotchas and Common Mistakes

### 7.1 spikes shape is `(n_electrodes, n_time)` — NOT sklearn-ready
The `spikes` variable is stored electrode-first. You MUST transpose before feeding to sklearn:
```python
spikes = ds["spikes"].values    # (n_electrodes, n_time)
X = spikes.T                    # (n_time, n_electrodes) ← sklearn wants this
```
`compute_binned_counts()` handles this internally — do not double-transpose.

### 7.2 Wiener filter trims both X and y
When `n_lags > 0`, `_build_lag_matrix(X, n_lags)` returns `(n - n_lags, f*(n_lags+1))`.
The corresponding `y` must be trimmed to `y[n_lags:]`. `DecodingPipeline._trim_y()` handles this internally. But if you use the raw `_build_lag_matrix()` outside the class, you must trim `y` manually.

### 7.3 `combined_zarr.keys()` returns empty on S3
Always list sessions via:
```python
root.group_keys()  # from zarr.open_group(s3_store, mode='r')
# or
data_utils.filter_sessions(filter_by="dataset_id", filter_value="DANDI_00070")
```

### 7.4 `trial_id == 0` vs `trial_id == -1`
V8 format uses `0` for inter-trial. Old code using `trial_id != -1` is wrong. Always use `trial_id != 0`.

### 7.5 Zenodo_3854034 has no discrete direction structure
`task_type == 'continuous_random'` — `trial_phase` is always 2. Calling `compute_direction_labels()` on this dataset raises `ValueError` by design. Filter before calling.

### 7.6 PCA `n_components` must be ≤ min(n_samples, n_features)
For small trial counts (Zenodo ~563, DANDI_000688 ~180), `n_components_pca=10` may exceed `n_samples`. `DecodingPipeline._prepare_X()` guards against this with `min(n_components_pca, X.shape[0], X.shape[1])`.

### 7.7 bin_phase sampling for scatter plots
`trial_phase` is at 1ms resolution. To get one phase label per 50ms bin:
```python
bin_phases = trial_phase_full[bin_size_ms // 2 :: bin_size_ms][:n_bins]
```
This samples the midpoint of each bin — more representative than the first or last sample.

---

## 8. Notebook Style Reference

Copy the header pattern from `demos/Phase2_SpikeAnalysis_Visualization_CV_1.ipynb`:
- First cell is a markdown cell with: `# Phase 3 — [Name]`, description, structure table
- Import cell confirms success with `print("✓ All imports successful")`
- S3 cell confirms with `print("✓ Connected to S3")`
- Each analysis section starts with a markdown cell explaining the WHY
- Visualization functions follow the `ax = None` pattern (can be called standalone or into existing axes)

---

## 9. Git Commit Convention

After implementation, commit with:
```
feat(decoding): add Phase 3 feature extraction and DecodingPipeline

- compute_binned_counts, smooth_firing_rates, compute_trial_features, compute_direction_labels
- DecodingPipeline (sklearn-compatible, ridge/wiener/lda/svm)
- Phase3_FeatureExtraction_CV.ipynb (Week 7 deliverable)
- Phase3_Decoding_Demo_CV.ipynb (Week 10 scaffold)
```

Branch: `docs/phase1-CamilaV-updates` — never push to `main` directly.

---

## 10. Quality Checklist

- [ ] All files created: `decoding/__init__.py`, `decoding/feature_extraction.py`, `decoding/decoding_module.py`, `decoding/notebooks/Phase3_FeatureExtraction_CV.ipynb`, `decoding/notebooks/Phase3_Decoding_Demo_CV.ipynb`
- [ ] Gate 1 import check passes (no ImportError)
- [ ] Gate 2 synthetic data test passes (all assertions green)
- [ ] `spikes.T` applied correctly inside `compute_binned_counts` (output shape is `(n_bins, n_electrodes)`)
- [ ] `shuffle=False` in `evaluate_cv` — verified by inspection
- [ ] `compute_direction_labels` raises `ValueError` for non-center_out sessions
- [ ] `DecodingPipeline` stores fitted `scaler_` and `pca_` as instance attributes
- [ ] Wiener filter trims y by `n_lags` rows internally
- [ ] All public functions have numpy-style docstrings with Parameters/Returns sections
- [ ] `from __future__ import annotations` at top of each `.py` file
- [ ] No hardcoded session IDs, dataset names, or S3 paths in `decoding/` package files

---

**Score: 8/10** — High confidence for one-pass implementation. The two open risks are: (1) direction label quality on real data (depends on position normalization consistency across datasets — validated approach for center-out), and (2) whether the Jupyter notebook JSON format is created correctly (use `nbformat` library or handcraft the minimal structure). If validation gates pass on synthetic data, real-data testing should be straightforward.
