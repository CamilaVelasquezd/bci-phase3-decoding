"""Utilities for discrete decoding — trial averaging and direction label computation. Not used for continuous decoders."""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import stats

from decoding.dim_reduction import compute_bin_phases, compute_binned_counts


def compute_binned_trial_ids(ds: xr.Dataset, bin_size_ms: int = 50) -> np.ndarray:
    """Bin trial_id from 1kHz timestep resolution to bin resolution using mode.

    A bin is assigned to whichever trial ID occupies the majority of that window.

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset with ds["trial_id"] of shape (n_time,) at 1kHz, int16.
    bin_size_ms : int
        Bin size in milliseconds. Must match the bin size used in compute_binned_counts.

    Returns
    -------
    np.ndarray
        Shape (n_bins,), int16. Trial ID per bin at bin resolution.

    Notes
    -----
    Trial ID convention in the zarr schema:
    - Positive IDs (e.g. 1–212): active reach trials
    - Negative IDs: inter-trial intervals linked to a nearby trial
    - ID = 0: transitions or padding

    Use active_mask = trial_id_binned > 0 to select only reach bins.
    This mask replaces the previous active_mask = bin_phases > 0 convention
    and is required for Leave-One-Trial-Out (LOTO) cross-validation.
    """
    trial_id_raw = ds["trial_id"].values  # (n_time,), int16
    n_time = len(trial_id_raw)
    n_bins = n_time // bin_size_ms

    # Trim to fit exactly into bins
    n_time_trimmed = n_bins * bin_size_ms
    trial_id_trimmed = trial_id_raw[:n_time_trimmed]

    # Reshape to (n_bins, bin_size_ms)
    trial_id_reshaped = trial_id_trimmed.reshape(n_bins, bin_size_ms)

    # Compute mode along axis=1 (for each bin, find the most common trial ID)
    trial_id_binned = stats.mode(trial_id_reshaped, axis=1, keepdims=False).mode

    return trial_id_binned.astype(np.int16)


def get_trial_data(
    X_all: np.ndarray,
    y_all: np.ndarray,
    trial_id_binned: np.ndarray,
) -> list[dict]:
    """Organize binned spike counts and velocity into per-trial chunks.

    Parameters
    ----------
    X_all : np.ndarray
        Shape (n_bins, n_electrodes). Full session binned spike counts.
    y_all : np.ndarray
        Shape (n_bins, 2). Full session binned velocity (vx, vy).
    trial_id_binned : np.ndarray
        Shape (n_bins,). Trial ID per bin from compute_binned_trial_ids.
        Positive IDs = active reach trials. Zero and negative = ITI/padding.

    Returns
    -------
    list[dict]
        One dict per trial, sorted by trial_id, each with keys:
            'trial_id' : int
            'X'        : np.ndarray, shape (n_bins_in_trial, n_electrodes)
            'y'        : np.ndarray, shape (n_bins_in_trial, 2)

    Notes
    -----
    Only positive trial IDs are included (active reach trials).
    Trials are sorted by trial_id to ensure consistent ordering across sessions.
    """
    unique_ids = np.unique(trial_id_binned[trial_id_binned > 0])
    trials = []
    for tid in unique_ids:
        mask = trial_id_binned == tid
        trials.append({
            "trial_id": int(tid),
            "X": X_all[mask],
            "y": y_all[mask],
        })
    return trials


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
        Shape ``(n_valid_trials, n_electrodes)``, float64. One row per trial
        with at least one reach-phase bin.
    valid_trial_ids : np.ndarray
        Shape ``(n_valid_trials,)``, int16. Trial IDs corresponding to rows of
        ``X_trials``. Use these to align direction labels.
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

    Uses mean velocity during the reach phase (``trial_phase == 2``) to
    determine the reach angle, then discretizes to the nearest of
    ``n_directions`` evenly spaced directions.

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset. ``task_type`` must be ``'center_out'`` or
        ``'center_out_maze'``.
    n_directions : int
        Number of target directions. Default 8 (standard center-out).

    Returns
    -------
    np.ndarray
        Shape ``(n_trials,)``, dtype int32. Values in
        ``[0, n_directions-1]``. Trials with no reach-phase samples are
        assigned -1.

    Raises
    ------
    ValueError
        If ``task_type`` is not ``'center_out'`` or ``'center_out_maze'``.
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


def compute_direction_labels_from_position(
    ds: xr.Dataset, n_directions: int = 8
) -> np.ndarray:
    """Compute discrete reach-direction label using cursor position displacement.

    Uses the displacement of cursor position during the reach phase
    (``trial_phase == 2``) rather than velocity. Prefer this function for
    DANDI_00070 and DANDI_000140 where vx == vy in every session
    (duplication bug).

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset. ``task_type`` must be ``'center_out'`` or
        ``'center_out_maze'``.
    n_directions : int
        Number of target directions. Default 8 (standard center-out).

    Returns
    -------
    np.ndarray
        Shape ``(n_trials,)``, dtype int32. Values in
        ``[0, n_directions-1]``. Trials with fewer than 2 reach-phase
        samples are assigned -1.

    Raises
    ------
    ValueError
        If ``task_type`` is not ``'center_out'`` or ``'center_out_maze'``.
    """
    task_type = ds.attrs["task_type"]
    if task_type not in ("center_out", "center_out_maze"):
        raise ValueError(
            f"compute_direction_labels_from_position requires center_out task, "
            f"got '{task_type}'. Do not call on Zenodo_3854034 (continuous_random)."
        )

    position = ds["position"].values        # (n_time, 2), float32
    trial_id = ds["trial_id"].values        # (n_time,), int16
    trial_phase = ds["trial_phase"].values  # (n_time,), int8

    all_trial_ids = np.unique(trial_id[trial_id > 0])
    bin_width = 360.0 / n_directions
    directions = []

    for tid in all_trial_ids:
        reach_mask = (trial_id == tid) & (trial_phase == 2)
        if reach_mask.sum() < 2:
            directions.append(-1)
            continue
        pos_reach = position[reach_mask]        # (n_reach, 2)
        dx = float(pos_reach[-1, 0] - pos_reach[0, 0])
        dy = float(pos_reach[-1, 1] - pos_reach[0, 1])
        angle = np.degrees(np.arctan2(dy, dx)) % 360.0
        direction_idx = int(np.round(angle / bin_width) % n_directions)
        directions.append(direction_idx)

    return np.array(directions, dtype=np.int32)

def compute_velocity_labels_binned(
    ds: xr.Dataset,
    stat_thresh: float = 0.03,
    fast_thresh: float = 0.15,
    n_directions: int = 8,
) -> np.ndarray:
    """Discretize cursor velocity into 17 classes at 1kHz bin resolution.

    Classes:
        0         → stationary (speed < stat_thresh)
        1–8       → slow directional (stat_thresh <= speed < fast_thresh)
        9–16      → fast directional (speed >= fast_thresh)

    Direction is computed as arctan2(vy, vx) quantized to the nearest
    of n_directions evenly spaced angles (0=E, 1=NE, ..., 7=SE).

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset with ds["velocity"] of shape (n_time, 2).
    stat_thresh : float
        Speed threshold below which a bin is considered stationary.
    fast_thresh : float
        Speed threshold above which a bin is considered fast.
    n_directions : int
        Number of directions. Default 8.

    Returns
    -------
    np.ndarray
        Shape (n_time,), int32. Discrete velocity label per timestep.
    """
    velocity = ds["velocity"].values  # (n_time, 2)
    vx = velocity[:, 0]
    vy = velocity[:, 1]
    speed = np.sqrt(vx**2 + vy**2)

    # Magnitude class: 0=stationary, 1=slow, 2=fast
    magnitude = np.where(speed < stat_thresh, 0,
                np.where(speed < fast_thresh, 1, 2))

    # Direction: quantize arctan2 to nearest multiple of pi/4
    angle = np.arctan2(vy, vx) % (2 * np.pi)
    direction = np.round(angle / (2 * np.pi / n_directions)).astype(int) % n_directions

    # Combine into 17 classes
    labels = np.where(magnitude == 0, 0,
             np.where(magnitude == 1, direction + 1, direction + n_directions + 1))

    return labels.astype(np.int32)