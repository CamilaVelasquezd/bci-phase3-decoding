from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def compute_binned_counts(
    ds: xr.Dataset,
    bin_size_ms: int = 50,
    bins_per_chunk: int = 500,
) -> np.ndarray:
    """Bin spike trains into count matrices.

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset with ``'spikes'`` variable of shape
        ``(n_electrodes, n_time)``.
    bin_size_ms : int
        Bin width in milliseconds. Default 50.
    bins_per_chunk : int
        Number of bins to read from the backing store at once. Keeping this
        bounded avoids materializing the full S3/Zarr spike array in memory.

    Returns
    -------
    np.ndarray
        Shape ``(n_bins, n_electrodes)``, dtype float64.
        Rows are time bins; columns are electrodes. sklearn-ready.
    """
    n_electrodes, n_time = ds["spikes"].shape
    n_bins = n_time // bin_size_ms
    if n_bins == 0:
        return np.empty((0, n_electrodes), dtype=np.float64)

    if bins_per_chunk <= 0:
        raise ValueError("bins_per_chunk must be positive")

    binned = np.empty((n_bins, n_electrodes), dtype=np.float64)
    for bin_start in range(0, n_bins, bins_per_chunk):
        bin_stop = min(bin_start + bins_per_chunk, n_bins)
        time_start = bin_start * bin_size_ms
        time_stop = bin_stop * bin_size_ms

        spikes_chunk = ds["spikes"].isel(time=slice(time_start, time_stop)).values
        chunk_bins = bin_stop - bin_start
        counts = spikes_chunk.reshape(n_electrodes, chunk_bins, bin_size_ms).sum(axis=2)
        binned[bin_start:bin_stop] = counts.T

    return binned


def compute_bin_phases(ds: xr.Dataset, bin_size_ms: int = 50) -> np.ndarray:
    """Return trial_phase label at the midpoint of each time bin.

    Parameters
    ----------
    ds : xr.Dataset
        Session dataset with ``'trial_phase'`` variable of shape ``(n_time,)``.
    bin_size_ms : int
        Bin width in milliseconds. Must match the value used in
        ``compute_binned_counts``.

    Returns
    -------
    np.ndarray
        Shape ``(n_bins,)``, dtype int8.
        Values: 0=inter, 1=pre-reach, 2=reach, 3=post.
    """
    trial_phase = ds["trial_phase"].values           # (n_time,)
    n_time = ds["spikes"].shape[1]
    n_bins = n_time // bin_size_ms
    return trial_phase[bin_size_ms // 2 :: bin_size_ms][:n_bins]


_VALID_METHODS = frozenset({"pca", "ica", "lda"})


class DimReducer:
    """Dimensionality reducer wrapping PCA, ICA, and LDA with a unified interface.

    Parameters
    ----------
    method : str
        One of ``'pca'``, ``'ica'``, ``'lda'``.
    n_components : int
        Number of output components. Clamped at fit time to
        ``min(n_components, n_samples - 1, n_features)``.
        For LDA additionally capped at ``n_classes - 1``.
    """

    def __init__(self, method: str = "pca", n_components: int = 10) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(_VALID_METHODS)}, got '{method}'"
            )
        self.method = method
        self.n_components = n_components
        self.model_: PCA | FastICA | LinearDiscriminantAnalysis | None = None
        self.scaler_: StandardScaler | None = None
        self._is_fitted: bool = False

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit scaler and reduction model on X, then return the reduced representation.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_trials, n_features). Feature matrix (e.g. output of
            ``compute_trial_averages``).
        y : np.ndarray or None
            Shape (n_trials,). Reach-direction labels. Required when
            ``method='lda'``; ignored otherwise.

        Returns
        -------
        np.ndarray
            Shape (n_trials, n_components), float64.

        Raises
        ------
        ValueError
            If ``method='lda'`` and y is None.
        """
        if self.method == "lda" and y is None:
            raise ValueError("y (direction labels) is required when method='lda'")

        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1]) 
        #No puede pedir más componentes que muestras o que features.

        if self.method == "lda":
            n_classes = len(np.unique(y))
            n_comp = min(n_comp, n_classes - 1)
            #LDA nunca puede dar más de n_clases - 1 componentes. Con 8 direcciones, 
            # máximo 7. 

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        #Normalizamos 
    

        if self.method == "pca":
            self.model_ = PCA(n_components=n_comp)
            X_reduced = self.model_.fit_transform(X_scaled)
        elif self.method == "ica":
            self.model_ = FastICA(n_components=n_comp, random_state=0, max_iter=500)
            X_reduced = self.model_.fit_transform(X_scaled)
        else:
            self.model_ = LinearDiscriminantAnalysis(n_components=n_comp)
            X_reduced = self.model_.fit_transform(X_scaled, y)

        self.n_components_actual_ = X_reduced.shape[1]
        if self.n_components_actual_ < self.n_components:
            warnings.warn(
                f"DimReducer(method='{self.method}'): requested {self.n_components} "
                f"components but only {self.n_components_actual_} were produced. "
                "For LDA this happens when n_unique_classes - 1 < n_components.",
                UserWarning,
                stacklevel=2,
            )

        self._is_fitted = True
        return X_reduced.astype(np.float64)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler and model to new data.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_trials, n_features). Must have the same features as the
            data passed to ``fit_transform``.

        Returns
        -------
        np.ndarray
            Shape (n_trials, n_components), float64.

        Raises
        ------
        RuntimeError
            If called before ``fit_transform``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "DimReducer is not fitted. Call fit_transform before transform."
            )
        X_scaled = self.scaler_.transform(X)
        return self.model_.transform(X_scaled).astype(np.float64)

    @property
    def explained_variance_ratio_(self) -> np.ndarray | None:
        """Explained variance ratio per component.

        Returns
        -------
        np.ndarray or None
            ``model_.explained_variance_ratio_`` for PCA and LDA; None for ICA.
        """
        if self.method in ("pca", "lda"):
            return self.model_.explained_variance_ratio_
        return None




