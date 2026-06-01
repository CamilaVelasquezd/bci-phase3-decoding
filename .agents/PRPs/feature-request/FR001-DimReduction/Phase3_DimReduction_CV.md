# PRP — Phase3_DimReduction_CV.ipynb

**Feature:** Week 1 dimensionality reduction notebook  
**Deliverables:** `decoding/feature_extraction.py` + `notebooks/Phase3_DimReduction_CV.ipynb`  
**Confidence score:** 9/10

---

## 0. Read CONTEXT.md first

Before writing a single line of code, read `CONTEXT.md` in full. Every decision in this PRP is grounded in it. Pay particular attention to:
- Section 3 (DatasetLoader API — use `DatasetLoader`, NOT `Combined_Dataset_Utils`)
- Section 5 (V8 schema — shapes, dtypes, trial_id convention)
- Section 8 (coding conventions — `shuffle=False`, no hardcoding, `from __future__ import annotations`)
- Section 10 (critical gotchas — spikes transpose, `trial_id == 0`, PCA guard, bin phase sampling)
- Section 12 (notebook style — cell structure, print checkpoints)

> **API warning:** The `.claude/skills/data-loader/SKILL.md` file uses the OLD `Combined_Dataset_Utils` API. That file is outdated. Always use `DatasetLoader` as shown in CONTEXT.md section 3.

---

## 1. What this feature is

A Jupyter notebook that explores dimensionality reduction on motor cortex spike data. It compares three methods — PCA (unsupervised), ICA (unsupervised), and LDA (supervised) — using spike binned counts from a single DANDI_00070 session. Outputs include a scree plot, scatter plots colored by trial phase and reach direction, and a quantitative comparison table (reconstruction error + silhouette score).

This is the **Week 1 deliverable** for the Solzbacher Lab Phase 3 BCI internship.

---

## 2. Scope: two deliverables

### Deliverable A — `decoding/feature_extraction.py`

This module must exist before the notebook can run. It contains four public functions:

| Function | Inputs | Output | Notes |
|---|---|---|---|
| `compute_binned_counts` | `ds, bin_size_ms=50` | `(n_bins, n_electrodes)` float64 | Transpose handled internally |
| `compute_bin_phases` | `ds, bin_size_ms=50` | `(n_bins,)` int8 | Midpoint sampling — see gotcha |
| `compute_trial_averages` | `ds, bin_size_ms=50` | `(X_trials, valid_trial_ids)` | Reach-phase avg per trial |
| `compute_direction_labels` | `ds, n_directions=8` | `(n_trials,)` int | Raises on wrong task_type |

Also create `decoding/__init__.py` (empty or with explicit imports).

### Deliverable B — `notebooks/Phase3_DimReduction_CV.ipynb`

A notebook with these sections (also create the `notebooks/` directory):

| # | Section | What it produces |
|---|---|---|
| — | Imports | `print("✓ All imports successful")` |
| — | S3 connection | `print("✓ Connected to S3")` |
| — | Session loading | Quick session summary |
| 1 | Feature extraction | `X_bins`, `X_trials`, `bin_phases`, `direction_labels` |
| 2 | PCA | Scree plot + phase scatter + direction scatter |
| 3 | ICA | Phase scatter + direction scatter |
| 4 | LDA | Direction scatter (supervised) |
| 5 | Comparison | DataFrame with reconstruction error + silhouette score |

---

## 3. V8 schema reference (from CONTEXT.md)

```
ds["spikes"]      → (n_electrodes, n_time),  uint8   ← ROWS are electrodes
ds["position"]    → (n_time, 2),             float32
ds["velocity"]    → (n_time, 2),             float32
ds["trial_id"]    → (n_time,),               int16   ← 0=inter-trial, +N=success, -N=failed
ds["trial_phase"] → (n_time,),               int8    ← 0=inter, 1=pre-reach, 2=reach, 3=post
ds.attrs["task_type"]     → str
ds.attrs["dataset_id"]    → str
ds.attrs["sampling_rate"] → 1000.0 (always)
```

**trial_id convention (V8):** Inter-trial is `== 0`, NOT `== -1`. Successful trials: `trial_id > 0`.

---

## 4. Dataset to use

**Use `DANDI_00070`** — best choice for this notebook:
- `task_type = "center_out"` → has clean direction structure (8 targets)
- ~191 electrodes, ~3000 successful trials
- Works for both phase-colored (bins) and direction-colored (trial-averaged) scatter

Load **only the first session** dynamically — never hardcode a session ID:
```python
session_id = loader.filter_sessions("dataset_id", "DANDI_00070")[0]
```

Do NOT use `DANDI_000140` (velocity bug). Do NOT use `Zenodo_3854034` for direction labels (no discrete trials).

---

## 5. S3 connection (copy exactly from CONTEXT.md)

```python
import configparser
import os
from bci_decoding_dataset import DatasetLoader

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
print("✓ Connected to S3")
```

---

## 6. Implementation blueprint

### 6.1 `decoding/feature_extraction.py` — full spec

```python
from __future__ import annotations

import numpy as np
import xarray as xr


def compute_binned_counts(ds: xr.Dataset, bin_size_ms: int = 50) -> np.ndarray:
    """Bin spike trains into count matrices.

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset with 'spikes' variable of shape (n_electrodes, n_time).
    bin_size_ms : int
        Bin width in milliseconds. Default 50.

    Returns
    -------
    np.ndarray
        Shape (n_bins, n_electrodes), dtype float64.
        Rows are time bins; columns are electrodes. sklearn-ready.
    """
    spikes = ds["spikes"].values          # (n_electrodes, n_time), uint8
    n_electrodes, n_time = spikes.shape
    n_bins = n_time // bin_size_ms
    trimmed = spikes[:, : n_bins * bin_size_ms]  # exact multiple
    # reshape: (n_electrodes, n_bins, bin_size_ms) → sum over last axis
    binned = trimmed.reshape(n_electrodes, n_bins, bin_size_ms).sum(axis=2)
    return binned.T.astype(np.float64)    # (n_bins, n_electrodes)


def compute_bin_phases(ds: xr.Dataset, bin_size_ms: int = 50) -> np.ndarray:
    """Return trial_phase label at the midpoint of each time bin.

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset with 'trial_phase' variable of shape (n_time,).
    bin_size_ms : int
        Bin width in milliseconds. Must match the value used in compute_binned_counts.

    Returns
    -------
    np.ndarray
        Shape (n_bins,), dtype int8. Values: 0=inter, 1=pre-reach, 2=reach, 3=post.
    """
    trial_phase = ds["trial_phase"].values           # (n_time,)
    n_time = ds["spikes"].shape[1]
    n_bins = n_time // bin_size_ms
    # sample at bin midpoint — critical gotcha from CONTEXT.md section 10
    return trial_phase[bin_size_ms // 2 :: bin_size_ms][:n_bins]


def compute_trial_averages(
    ds: xr.Dataset, bin_size_ms: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean binned spike counts during the reach phase for each trial.

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset.
    bin_size_ms : int
        Bin width in milliseconds.

    Returns
    -------
    X_trials : np.ndarray
        Shape (n_valid_trials, n_electrodes), float64. One row per trial with
        ≥1 reach-phase bin.
    valid_trial_ids : np.ndarray
        Shape (n_valid_trials,), int16. Trial IDs corresponding to X_trials rows.
        Use these to align direction labels.
    """
    X_bins = compute_binned_counts(ds, bin_size_ms)          # (n_bins, n_el)
    bin_phases = compute_bin_phases(ds, bin_size_ms)          # (n_bins,)

    trial_id_full = ds["trial_id"].values                     # (n_time,)
    n_bins = X_bins.shape[0]
    bin_trial_ids = trial_id_full[bin_size_ms // 2 :: bin_size_ms][:n_bins]

    all_trial_ids = np.unique(trial_id_full[trial_id_full > 0])

    X_list, valid_ids = [], []
    for tid in all_trial_ids:
        mask = (bin_trial_ids == tid) & (bin_phases == 2)
        if mask.sum() == 0:
            continue
        X_list.append(X_bins[mask].mean(axis=0))
        valid_ids.append(tid)

    return np.array(X_list, dtype=np.float64), np.array(valid_ids, dtype=np.int16)


def compute_direction_labels(ds: xr.Dataset, n_directions: int = 8) -> np.ndarray:
    """Compute discrete reach-direction label for each successful trial.

    Uses mean velocity during the reach phase (trial_phase == 2) to determine
    the reach angle, then discretizes to the nearest of n_directions evenly
    spaced directions.

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset. task_type must be 'center_out' or 'center_out_maze'.
    n_directions : int
        Number of target directions. Default 8 (standard center-out).

    Returns
    -------
    np.ndarray
        Shape (n_trials,), dtype int32. Values in [0, n_directions-1].
        Trials with no reach-phase samples are assigned -1.

    Raises
    ------
    ValueError
        If task_type is not 'center_out' or 'center_out_maze'.
    """
    task_type = ds.attrs["task_type"]
    if task_type not in ("center_out", "center_out_maze"):
        raise ValueError(
            f"compute_direction_labels requires center_out task, got '{task_type}'. "
            "Do not call on Zenodo_3854034 (continuous_random)."
        )

    velocity = ds["velocity"].values       # (n_time, 2), float32
    trial_id = ds["trial_id"].values       # (n_time,), int16
    trial_phase = ds["trial_phase"].values # (n_time,), int8

    all_trial_ids = np.unique(trial_id[trial_id > 0])
    bin_width = 360.0 / n_directions
    directions = []

    for tid in all_trial_ids:
        reach_mask = (trial_id == tid) & (trial_phase == 2)
        if reach_mask.sum() == 0:
            directions.append(-1)
            continue
        vx = velocity[reach_mask, 0].mean()
        vy = velocity[reach_mask, 1].mean()
        angle = np.degrees(np.arctan2(vy, vx)) % 360.0
        direction_idx = int(np.round(angle / bin_width) % n_directions)
        directions.append(direction_idx)

    return np.array(directions, dtype=np.int32)
```

### 6.2 `decoding/__init__.py`

```python
from __future__ import annotations
```

### 6.3 Notebook cell structure

Below is the **exact cell-by-cell plan** for the notebook. The implementing agent must follow this structure precisely.

---

**Cell 1 — Markdown (title)**

```markdown
# Phase 3 — Dimensionality Reduction

Comparison of PCA, ICA, and LDA applied to binned motor cortex spike counts.

| Section | Content |
|---|---|
| 1 | Feature extraction — binned spike counts |
| 2 | PCA — scree plot + scatter by trial phase + scatter by direction |
| 3 | ICA — scatter by trial phase + scatter by direction |
| 4 | LDA — scatter by direction (supervised) |
| 5 | Comparison table — reconstruction error + silhouette score |

**Dataset:** DANDI_00070 (center_out, ~191 electrodes, ~3000 trials)
**Bin size:** 50 ms
```

---

**Cell 2 — Imports**

```python
from __future__ import annotations

import configparser
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from bci_decoding_dataset import DatasetLoader
from decoding.feature_extraction import (
    compute_bin_phases,
    compute_binned_counts,
    compute_direction_labels,
    compute_trial_averages,
)

print("✓ All imports successful")
```

---

**Cell 3 — S3 connection**

Exact pattern from CONTEXT.md section 3.2. Ends with:
```python
print("✓ Connected to S3")
```

---

**Cell 4 — Session loading**

```python
# Dynamically pick first DANDI_00070 session — never hardcode a session ID
session_id = loader.filter_sessions("dataset_id", "DANDI_00070")[0]
ds = loader.get_processed_data_from_session(session_id)

print(f"Session:    {session_id}")
print(f"Dataset:    {ds.attrs['dataset_id']}")
print(f"Task type:  {ds.attrs['task_type']}")
print(f"Electrodes: {ds['spikes'].shape[0]}")
print(f"Duration:   {ds['spikes'].shape[1]} ms ({ds['spikes'].shape[1]/1000:.1f} s)")

trial_id_vals = ds["trial_id"].values
n_trials = len(np.unique(trial_id_vals[trial_id_vals > 0]))
print(f"Trials:     {n_trials} successful")
```

---

**Cell 5 — Markdown: Section 1 — Feature Extraction**

Explain WHY we bin: rate coding hypothesis, reduce 1ms resolution to 50ms "firing rate" windows, output is (n_bins, n_electrodes) matrix for sklearn.

---

**Cell 6 — Feature extraction code**

```python
BIN_SIZE_MS = 50
N_COMPONENTS = 10   # for PCA and ICA; guarded below

# Bin-level features (for phase scatter)
X_bins = compute_binned_counts(ds, bin_size_ms=BIN_SIZE_MS)  # (n_bins, n_electrodes)
bin_phases = compute_bin_phases(ds, bin_size_ms=BIN_SIZE_MS)  # (n_bins,)

# Trial-level features (for direction scatter + comparison table)
X_trials, valid_trial_ids = compute_trial_averages(ds, bin_size_ms=BIN_SIZE_MS)

# Direction labels aligned to valid_trial_ids
all_trial_ids = np.unique(ds["trial_id"].values[ds["trial_id"].values > 0])
all_dir_labels = compute_direction_labels(ds)
trial_to_dir = dict(zip(all_trial_ids, all_dir_labels))
direction_labels = np.array([trial_to_dir[tid] for tid in valid_trial_ids])

# Filter out trials with no reach phase (direction == -1)
valid_mask = direction_labels >= 0
X_trials = X_trials[valid_mask]
direction_labels = direction_labels[valid_mask]

# Scale — fit on each representation separately
scaler_bins = StandardScaler()
X_bins_scaled = scaler_bins.fit_transform(X_bins)

scaler_trials = StandardScaler()
X_trials_scaled = scaler_trials.fit_transform(X_trials)

print(f"X_bins shape:   {X_bins.shape}")
print(f"X_trials shape: {X_trials.shape}")
print(f"Directions:     {np.unique(direction_labels)}")
```

---

**Cell 7 — Markdown: Section 2 — PCA**

Explain WHY: linear projection maximizing variance. No assumptions about independence (unlike ICA) or class labels (unlike LDA). Scree plot tells us how many components capture meaningful variance.

---

**Cell 8 — PCA fit**

```python
# Guard n_components against small datasets (critical for DANDI_000688)
n_pca = min(N_COMPONENTS, X_trials_scaled.shape[0] - 1, X_trials_scaled.shape[1])
n_pca_bins = min(20, X_bins_scaled.shape[0] - 1, X_bins_scaled.shape[1])

# Fit on trial-level (comparison) and bin-level (scree + phase scatter)
pca_trials = PCA(n_components=n_pca, random_state=42)
X_pca_trials = pca_trials.fit_transform(X_trials_scaled)   # (n_trials, n_pca)

pca_bins = PCA(n_components=n_pca_bins, random_state=42)
X_pca_bins = pca_bins.fit_transform(X_bins_scaled)          # (n_bins, n_pca_bins)

print(f"PCA (trial-level): {n_pca} components, "
      f"cumulative var = {pca_trials.explained_variance_ratio_.sum():.1%}")
```

---

**Cell 9 — Scree plot**

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: bar chart of explained variance ratio
ax = axes[0]
ax.bar(range(1, n_pca_bins + 1), pca_bins.explained_variance_ratio_, color="steelblue")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance Ratio")
ax.set_title("PCA Scree Plot")

# Right: cumulative explained variance
ax2 = axes[1]
cumulative = np.cumsum(pca_bins.explained_variance_ratio_)
ax2.plot(range(1, n_pca_bins + 1), cumulative, marker="o", color="steelblue")
ax2.axhline(0.90, color="red", linestyle="--", label="90% threshold")
ax2.set_xlabel("Number of Components")
ax2.set_ylabel("Cumulative Explained Variance")
ax2.set_title("Cumulative Explained Variance")
ax2.legend()

plt.tight_layout()
plt.show()
```

---

**Cell 10 — PC1 vs PC2 colored by trial phase (bin-level)**

```python
PHASE_COLORS = {0: "lightgray", 1: "steelblue", 2: "tomato", 3: "seagreen"}
PHASE_LABELS = {0: "Inter-trial", 1: "Pre-reach", 2: "Reach", 3: "Post-reach"}

# Subsample bins to avoid overplotting (max 3000 points)
np.random.seed(42)
n_plot = min(3000, len(X_pca_bins))
idx = np.random.choice(len(X_pca_bins), size=n_plot, replace=False)
idx_sorted = np.sort(idx)

fig, ax = plt.subplots(figsize=(7, 6))
for phase, color in PHASE_COLORS.items():
    mask = bin_phases[idx_sorted] == phase
    ax.scatter(
        X_pca_bins[idx_sorted[mask], 0],
        X_pca_bins[idx_sorted[mask], 1],
        c=color, label=PHASE_LABELS[phase], alpha=0.5, s=10
    )
ax.set_xlabel(f"PC1 ({pca_bins.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca_bins.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("PCA — PC1 vs PC2 (colored by trial phase)")
ax.legend(markerscale=2)
plt.tight_layout()
plt.show()
```

---

**Cell 11 — PC1 vs PC2 colored by reach direction (trial-level)**

```python
cmap = plt.cm.hsv
n_dir = len(np.unique(direction_labels))
colors_dir = [cmap(d / n_dir) for d in range(n_dir)]

fig, ax = plt.subplots(figsize=(7, 6))
for d in range(n_dir):
    mask = direction_labels == d
    angle_deg = d * (360 / n_dir)
    ax.scatter(
        X_pca_trials[mask, 0], X_pca_trials[mask, 1],
        c=[colors_dir[d]], label=f"{angle_deg:.0f}°", alpha=0.6, s=20
    )
ax.set_xlabel(f"PC1 ({pca_trials.explained_variance_ratio_[0]:.1%} var)")
ax.set_ylabel(f"PC2 ({pca_trials.explained_variance_ratio_[1]:.1%} var)")
ax.set_title("PCA — PC1 vs PC2 (colored by reach direction, trial-averaged)")
ax.legend(title="Direction", ncol=2, markerscale=2, fontsize=8)
plt.tight_layout()
plt.show()
```

---

**Cell 12 — Markdown: Section 3 — ICA**

Explain WHY: seeks statistically independent components (not just uncorrelated). Assumes spike sources are non-Gaussian. Compare to PCA — does it reveal clearer direction structure?

---

**Cell 13 — ICA fit**

```python
n_ica = min(N_COMPONENTS, X_trials_scaled.shape[0] - 1, X_trials_scaled.shape[1])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)  # convergence warning
    ica = FastICA(n_components=n_ica, max_iter=1000, tol=0.01, random_state=42)
    X_ica = ica.fit_transform(X_trials_scaled)   # (n_trials, n_ica)
# If ICA does not converge, increase max_iter or reduce n_components
# Convergence is dataset-dependent — inspect IC scatter visually
print(f"ICA: {n_ica} components fitted")
```

---

**Cell 14 — IC1 vs IC2 colored by trial phase**

For bin-level ICA phase scatter: fit a second ICA on `X_bins_scaled`, subsample same as PCA phase scatter.

```python
n_ica_bins = min(N_COMPONENTS, X_bins_scaled.shape[0] - 1, X_bins_scaled.shape[1])
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    ica_bins = FastICA(n_components=n_ica_bins, max_iter=1000, tol=0.01, random_state=42)
    X_ica_bins = ica_bins.fit_transform(X_bins_scaled)

fig, ax = plt.subplots(figsize=(7, 6))
for phase, color in PHASE_COLORS.items():
    mask = bin_phases[idx_sorted] == phase
    ax.scatter(
        X_ica_bins[idx_sorted[mask], 0],
        X_ica_bins[idx_sorted[mask], 1],
        c=color, label=PHASE_LABELS[phase], alpha=0.5, s=10
    )
ax.set_xlabel("IC1")
ax.set_ylabel("IC2")
ax.set_title("ICA — IC1 vs IC2 (colored by trial phase)")
ax.legend(markerscale=2)
plt.tight_layout()
plt.show()
```

---

**Cell 15 — IC1 vs IC2 colored by direction (trial-level)**

Same pattern as Cell 11 but with `X_ica` and no explained variance labels.

---

**Cell 16 — Markdown: Section 4 — LDA**

Explain WHY: supervised. Finds directions that maximize between-class / within-class variance ratio. Expected to produce the best direction separation (it's optimized for it). Max components = n_directions − 1 = 7.

---

**Cell 17 — LDA fit**

```python
n_lda = min(len(np.unique(direction_labels)) - 1, X_trials_scaled.shape[1])

lda = LinearDiscriminantAnalysis(n_components=n_lda)
X_lda = lda.fit_transform(X_trials_scaled, direction_labels)  # (n_trials, n_lda)

print(f"LDA: {n_lda} components")
print(f"Explained variance ratio: {lda.explained_variance_ratio_[:3]}")
```

---

**Cell 18 — LD1 vs LD2 colored by direction**

Same scatter pattern as Cells 11 and 15 but using `X_lda`. No phase scatter (LDA is supervised — it makes no sense to color by phase).

---

**Cell 19 — Markdown: Section 5 — Comparison**

Explain the three metrics:
- Reconstruction error: how well does the reduced representation preserve the original signal?
- Silhouette score: how well-separated are the 8 direction clusters in reduced space?
- LDA has no reconstruction error (no `inverse_transform` in sklearn) → "N/A"

---

**Cell 20 — Compute metrics**

```python
# PCA reconstruction error
X_pca_reconstructed = pca_trials.inverse_transform(X_pca_trials)
pca_recon = np.mean((X_trials_scaled - X_pca_reconstructed) ** 2)

# ICA reconstruction error
X_ica_reconstructed = ica.inverse_transform(X_ica)
ica_recon = np.mean((X_trials_scaled - X_ica_reconstructed) ** 2)

# Silhouette scores (all use trial-level direction labels)
pca_sil = silhouette_score(X_pca_trials, direction_labels)
ica_sil = silhouette_score(X_ica, direction_labels)
lda_sil = silhouette_score(X_lda, direction_labels)

results = pd.DataFrame({
    "Method": [f"PCA ({n_pca} components)", f"ICA ({n_ica} components)", f"LDA ({n_lda} components)"],
    "Reconstruction Error (MSE)": [f"{pca_recon:.4f}", f"{ica_recon:.4f}", "N/A (supervised)"],
    "Silhouette Score": [f"{pca_sil:.3f}", f"{ica_sil:.3f}", f"{lda_sil:.3f}"],
    "Notes": [
        f"Explains {pca_trials.explained_variance_ratio_.sum():.1%} variance",
        "Independent components",
        "Supervised — uses direction labels",
    ],
})
display(results)
```

---

## 7. Critical gotchas (all from CONTEXT.md section 10)

| Gotcha | What to do |
|---|---|
| `spikes` shape is `(n_electrodes, n_time)` | `compute_binned_counts` does `.T` internally — do NOT double-transpose |
| `trial_id == 0` is inter-trial | Use `trial_id > 0` for successful trials, NOT `trial_id != -1` |
| Bin phase midpoint sampling | `trial_phase[bin_size_ms // 2 :: bin_size_ms][:n_bins]` — implemented in `compute_bin_phases` |
| PCA `n_components` guard | `min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])` — critical for DANDI_000688 |
| `sp.pca()` returns a Figure | Do not call `sp.pca()` for the decoding pipeline — it exposes no components |
| `shuffle=False` | No train/test splits in this notebook — silhouette score uses all data |
| DANDI_000140 velocity bug | Not used in this notebook — noted for future reference |
| ICA convergence | `max_iter=1000`, `tol=0.01`, suppress with `warnings.catch_warnings` |
| LDA no `inverse_transform` | reconstruction error = "N/A" in comparison table — do NOT attempt `lda.inverse_transform` |
| Direction label alignment | `compute_trial_averages` returns `valid_trial_ids` → align via `trial_to_dir` dict |
| Scatter overplotting | Subsample bins to ≤ 3000 for phase scatter: `np.random.choice(len(X_pca_bins), 3000)` |

---

## 8. sklearn documentation URLs

- PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- FastICA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
- LDA: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
- Silhouette score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
- StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

**ICA gotcha from sklearn docs:** `FastICA.inverse_transform` uses the pseudoinverse of the mixing matrix. It exists in sklearn but is NOT numerically exact — expect slightly higher reconstruction error than PCA for the same n_components.

**LDA gotcha from sklearn docs:** `LinearDiscriminantAnalysis` does NOT have `inverse_transform`. Attempting it will raise `AttributeError`. Do not try.

---

## 9. Validation gates (run after implementing, before marking complete)

### Gate 1 — Package imports

```bash
conda run -n bci-ds python -c "
from decoding.feature_extraction import (
    compute_binned_counts, compute_bin_phases,
    compute_trial_averages, compute_direction_labels
)
print('Gate 1 passed: all imports OK')
"
```

### Gate 2 — `compute_binned_counts` shape contract

```bash
conda run -n bci-ds python -c "
import numpy as np, xarray as xr
from decoding.feature_extraction import compute_binned_counts

n_el, n_t = 96, 10000
spikes = xr.DataArray(
    np.random.randint(0, 2, (n_el, n_t), dtype=np.uint8),
    dims=['electrodes', 'time']
)
ds = xr.Dataset({'spikes': spikes})

X = compute_binned_counts(ds, bin_size_ms=50)
assert X.shape == (n_t // 50, n_el), f'Shape fail: {X.shape}'
assert X.dtype == np.float64, f'Dtype fail: {X.dtype}'
assert X.min() >= 0, 'Negative counts impossible'
print('Gate 2 passed: compute_binned_counts shape/dtype OK')
"
```

### Gate 3 — `compute_bin_phases` midpoint sampling

```bash
conda run -n bci-ds python -c "
import numpy as np, xarray as xr
from decoding.feature_extraction import compute_bin_phases, compute_binned_counts

n_el, n_t = 10, 1000
phase = np.zeros(n_t, dtype=np.int8)
phase[25:75] = 2   # reach phase from t=25 to t=75
spikes = xr.DataArray(np.zeros((n_el, n_t), dtype=np.uint8), dims=['electrodes','time'])
ds = xr.Dataset({
    'spikes': spikes,
    'trial_phase': xr.DataArray(phase, dims=['time']),
})
bp = compute_bin_phases(ds, bin_size_ms=50)
# Bin 0: t=0..49, midpoint t=25 → phase 2
assert bp[0] == 2, f'Expected 2, got {bp[0]}'
# Bin 1: t=50..99, midpoint t=75 → phase 2
assert bp[1] == 2, f'Expected 2, got {bp[1]}'
# Bin 2: t=100..149, midpoint t=125 → phase 0
assert bp[2] == 0, f'Expected 0, got {bp[2]}'
print('Gate 3 passed: compute_bin_phases midpoint sampling OK')
"
```

### Gate 4 — `compute_direction_labels` raises on wrong task_type

```bash
conda run -n bci-ds python -c "
import numpy as np, xarray as xr
from decoding.feature_extraction import compute_direction_labels

n_t = 100
ds = xr.Dataset(
    {
        'spikes': xr.DataArray(np.zeros((10, n_t), dtype=np.uint8), dims=['electrodes','time']),
        'velocity': xr.DataArray(np.zeros((n_t, 2), dtype=np.float32), dims=['time','xy']),
        'trial_id': xr.DataArray(np.zeros(n_t, dtype=np.int16), dims=['time']),
        'trial_phase': xr.DataArray(np.zeros(n_t, dtype=np.int8), dims=['time']),
    },
    attrs={'task_type': 'continuous_random', 'dataset_id': 'fake', 'sampling_rate': 1000.0}
)
try:
    compute_direction_labels(ds)
    raise AssertionError('Should have raised ValueError')
except ValueError as e:
    print(f'Gate 4 passed: ValueError raised — {e}')
"
```

### Gate 5 — Notebook is valid nbformat JSON

```bash
conda run -n bci-ds python -c "
import nbformat
with open('notebooks/Phase3_DimReduction_CV.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
nbformat.validate(nb)
print('Gate 5 passed: notebook nbformat valid')
"
```

---

## 10. Implementation order

Work in this order — each step depends on the previous:

1. **Create `decoding/__init__.py`** (empty `from __future__ import annotations` only)
2. **Create `decoding/feature_extraction.py`** with all four functions (section 6.1)
3. **Run Gates 1–4** to verify the module before touching the notebook
4. **Create `notebooks/` directory** (just create the file, the directory auto-creates)
5. **Create `notebooks/Phase3_DimReduction_CV.ipynb`** cell by cell following section 6.3
6. **Run Gate 5** to verify the notebook JSON
7. **Use the `decoding-validator` agent** to review both files before marking done

---

## 11. Coding conventions checklist

- [ ] `from __future__ import annotations` at top of `decoding/__init__.py` and `decoding/feature_extraction.py`
- [ ] Numpy-style docstrings (Parameters / Returns) on all four public functions
- [ ] No hardcoded session IDs — use `loader.filter_sessions("dataset_id", "DANDI_00070")[0]`
- [ ] No hardcoded S3 paths — all via `DatasetLoader`
- [ ] `shuffle=False` — no train/test split in this notebook; silhouette uses full data (note this in a markdown cell)
- [ ] `DatasetLoader` used (NOT `Combined_Dataset_Utils`)
- [ ] `trial_id == 0` is inter-trial, `trial_id > 0` for success
- [ ] `compute_direction_labels` only called after confirming `task_type` (done internally)
- [ ] PCA `n_components` guarded with `min(N, X.shape[0]-1, X.shape[1])`
- [ ] ICA fitted with `max_iter=1000, tol=0.01, random_state=42`
- [ ] LDA reconstruction error reported as "N/A" — no `inverse_transform` attempt
- [ ] Notebook Cell 2 ends with `print("✓ All imports successful")`
- [ ] Notebook Cell 3 ends with `print("✓ Connected to S3")`
- [ ] Each section starts with a markdown cell explaining WHY before code

---

## 12. Quality checklist

- [ ] CONTEXT.md reference included ✅
- [ ] S3 connection pattern copied exactly ✅
- [ ] V8 schema documented ✅
- [ ] Dataset quirks and known bugs noted ✅
- [ ] Validation gates are executable (conda run -n bci-ds) ✅
- [ ] shuffle=False addressed ✅
- [ ] No hardcoded paths or session IDs ✅
- [ ] sklearn interface respected ✅
- [ ] Clear implementation path with ordered tasks ✅

**Confidence score: 9/10** — The one remaining uncertainty is ICA convergence on a real session (solver behavior is dataset-dependent). The `max_iter=1000` + `tol=0.01` settings and the `warnings.catch_warnings` guard make this robust, but a human should visually inspect the IC scatter plot to confirm meaningful structure was found.
