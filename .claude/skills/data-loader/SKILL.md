---
name: data-loader
description: Loads and inspects BCI dataset sessions for Phase 3 decoding work. Use when selecting sessions to decode, checking which datasets are suitable for regression vs classification, verifying data shapes, or debugging S3 loading failures. Does not cover ingestion or health checks — only consumer-side loading.
allowed-tools: Bash(python *), Bash(conda *), Read, Glob, Grep
---

# Data Loader — Phase 3 Decoding

Reference for loading BCI sessions into feature matrices ready for sklearn.

## Dataset quick reference

| Dataset ID | Task type | Electrodes | Trials | Use for |
|---|---|---|---|---|
| `DANDI_00070` | `center_out` | ~191 | ~3000 | ✅ Regression + Classification (best) |
| `DANDI_000688` | `center_out` | ~54 | ~180 | ✅ Regression + Classification (small) |
| `Zenodo_3854034` | `continuous_random` | ~96 | — | ✅ Regression only — no discrete trials |
| `DANDI_000140` | `center_out_maze` | — | — | ⚠️ Skip velocity — vx/vy bug. Position OK. |

---

## S3 connection (copy exactly)

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
```

**WARNING:** `data_utils.combined_zarr.keys()` returns empty on S3. Always list sessions via:
```python
data_utils.filter_sessions(filter_by="dataset_id", filter_value="DANDI_00070")
# or
list(root.group_keys())
```

---

## Loading a session

```python
# Load a session
dandi_70 = data_utils.filter_sessions(filter_by="dataset_id", filter_value="DANDI_00070")
session_id = dandi_70[0]
ds = data_utils.get_processed_data_from_session(session_id)

# Load arrays into memory
spikes      = ds["spikes"].values         # (n_electrodes, n_time), uint8
position    = ds["position"].values       # (n_time, 2), float64
velocity    = ds["velocity"].values       # (n_time, 2), float64
trial_id    = ds["trial_id"].values       # (n_time,), int16
trial_phase = ds["trial_phase"].values    # (n_time,), int8

# Session metadata
print(ds.attrs["dataset_id"])    # e.g. 'DANDI_00070'
print(ds.attrs["task_type"])     # e.g. 'center_out'
print(ds.attrs["sampling_rate"]) # always 1000.0
```

---

## Trial masks

```python
active_mask          = trial_id != 0        # excludes inter-trial (V8: use 0, NOT -1)
successful_mask      = trial_id > 0         # successful trials only
successful_trial_ids = np.unique(trial_id[trial_id > 0])
```

---

## Shape check before sklearn

```python
# spikes is (n_electrodes, n_time) — NOT sklearn-ready
# Must transpose:
X_raw = spikes.T   # (n_time, n_electrodes) ← sklearn wants this

# compute_binned_counts() handles the transpose internally
# Do NOT double-transpose if using that function
from decoding.feature_extraction import compute_binned_counts
X = compute_binned_counts(ds, bin_size_ms=50)  # (n_bins, n_electrodes) ✅
```

---

## Quick session inspection

```python
import numpy as np

print(f"Dataset:      {ds.attrs['dataset_id']}")
print(f"Task type:    {ds.attrs['task_type']}")
print(f"Duration:     {spikes.shape[1]} ms ({spikes.shape[1]/1000:.1f}s)")
print(f"Electrodes:   {spikes.shape[0]}")
print(f"Trials (ok):  {len(np.unique(trial_id[trial_id > 0]))}")
print(f"Firing rate:  {spikes.mean()*1000:.2f} Hz avg")
print(f"Reach time:   {(trial_phase == 2).sum()} ms")
```

---

## Common issues

| Problem | Cause | Fix |
|---|---|---|
| `combined_zarr.keys()` empty | S3 limitation | Use `filter_sessions()` or `root.group_keys()` |
| Wrong trial mask | Old V7 code | Use `trial_id != 0`, not `trial_id != -1` |
| sklearn shape error | spikes not transposed | Use `spikes.T` or `compute_binned_counts()` |
| vx == vy | DANDI_000140 bug | Skip velocity from that dataset |
| PCA error on small dataset | n_components > n_samples | Guard: `min(n_components, X.shape[0], X.shape[1])` |
| `ValueError` from `compute_direction_labels` | Wrong task_type | Only call on `center_out` or `center_out_maze` |
