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
