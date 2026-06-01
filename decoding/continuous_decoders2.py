from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _r2_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute R² per velocity dimension.

    Parameters
    ----------
    y_true : np.ndarray
        Shape ``(n_samples, 2)``. Ground-truth velocity — column 0 = vx,
        column 1 = vy.
    y_pred : np.ndarray
        Shape ``(n_samples, 2)``. Predicted velocity.

    Returns
    -------
    dict[str, float]
        ``{'vx': R²_vx, 'vy': R²_vy}``.
    """
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    return {"vx": float(r2[0]), "vy": float(r2[1])}


# ---------------------------------------------------------------------------
# RidgeDecoder
# ---------------------------------------------------------------------------

class RidgeDecoder:
    """Regularized linear decoder for continuous kinematic targets.

    Wraps :class:`sklearn.linear_model.Ridge` with an internal
    :class:`~sklearn.preprocessing.StandardScaler`.  ``score`` returns R²
    separately for vx and vy instead of the sklearn default (mean R²).

    Parameters
    ----------
    alpha : float
        L2 regularization strength passed to ``Ridge``. Default 1.0.

    Attributes
    ----------
    model_ : Ridge
        Fitted Ridge regression model.
    scaler_ : StandardScaler
        Fitted zero-mean / unit-variance scaler.
    _is_fitted : bool
        True after ``fit`` has been called.

    Examples
    --------
    >>> dec = RidgeDecoder(alpha=1.0)
    >>> dec.fit(X_train, y_train)
    >>> dec.score(X_test, y_test)
    {'vx': 0.52, 'vy': 0.48}
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.model_: Ridge | None = None
        self.scaler_: StandardScaler | None = None
        self._is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeDecoder:
        """Fit scaler and Ridge model.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``. Neural feature matrix (e.g. output
            of ``compute_binned_counts`` or PCA-reduced counts).
        y : np.ndarray
            Shape ``(n_bins, 2)``. Velocity targets — column 0 = vx, 1 = vy —
            aligned to the same bins as ``X``.

        Returns
        -------
        RidgeDecoder
            self, for method chaining.
        """
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict velocity from neural features.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins, 2)``, float64.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RidgeDecoder is not fitted. Call fit before predict."
            )
        return self.model_.predict(
            self.scaler_.transform(X)
        ).astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute R² per velocity dimension.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y : np.ndarray
            Shape ``(n_bins, 2)``. Ground-truth velocity.

        Returns
        -------
        dict[str, float]
            ``{'vx': R²_vx, 'vy': R²_vy}``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        return _r2_dict(y, self.predict(X))


# ---------------------------------------------------------------------------
# WienerFilterDecoder
# ---------------------------------------------------------------------------

class WienerFilterDecoder:
    """Wiener filter: linear decoder with time-lagged spike bins.

    Appends the previous ``n_lags`` spike-count bins as extra features before
    fitting.  This gives the model temporal context — motor cortex activity
    leads hand velocity by ~100–200 ms, so temporal lags are the key advantage
    over plain Ridge.

    The lag matrix has ``n_bins - n_lags`` rows; ``fit`` and ``score`` trim
    ``y`` accordingly so the first decodable bin is at index ``n_lags``.

    Parameters
    ----------
    n_lags : int
        Number of past bins appended as features. Default 5.
        At ``bin_size_ms=50`` this is 250 ms of history.

    Attributes
    ----------
    model_ : LinearRegression
        Fitted OLS model.
    scaler_ : StandardScaler
        Fitted scaler applied to the lagged feature matrix.
    _is_fitted : bool
        True after ``fit`` has been called.

    Notes
    -----
    ``_build_lagged_features(X, n_lags)`` stacks the current bin alongside
    ``n_lags`` previous bins column-wise, yielding a matrix of shape
    ``(n_bins - n_lags, n_features * (n_lags + 1))``.

    When the input ``X`` is a concatenation of non-contiguous segments (e.g.
    active-phase bins from different trials), the lag construction will cross
    segment boundaries for the first ``n_lags`` rows of each segment.
    This is a known limitation of the flat-array Wiener filter approach.

    Examples
    --------
    >>> dec = WienerFilterDecoder(n_lags=5)
    >>> dec.fit(X_train, y_train)
    >>> dec.score(X_test, y_test)
    {'vx': 0.55, 'vy': 0.50}
    """

    def __init__(self, n_lags: int = 5) -> None:
        self.n_lags = n_lags
        self.model_: LinearRegression | None = None
        self.scaler_: StandardScaler | None = None
        self._is_fitted: bool = False

    def _build_lagged_features(self, X: np.ndarray, n_lags: int) -> np.ndarray:
        """Stack current and past bins into a single feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        n_lags : int
            Number of past bins to append.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins - n_lags, n_features * (n_lags + 1))``, float64.
            Row ``i`` concatenates ``X[i + n_lags]`` (lag 0) through
            ``X[i]`` (lag ``n_lags``).
        """
        n_bins, _ = X.shape
        rows = []
        for lag in range(n_lags + 1):
            start = n_lags - lag
            end = n_bins - lag if lag > 0 else None
            rows.append(X[start:end])
        return np.hstack(rows).astype(np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> WienerFilterDecoder:
        """Build lag matrix, fit scaler and LinearRegression.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y : np.ndarray
            Shape ``(n_bins, 2)``. Velocity targets aligned to ``X``.
            The first ``n_lags`` rows are trimmed internally to align with
            the lag matrix.

        Returns
        -------
        WienerFilterDecoder
            self, for method chaining.
        """
        X_lagged = self._build_lagged_features(X, self.n_lags)
        y_trimmed = y[self.n_lags:]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_lagged)
        self.model_ = LinearRegression()
        self.model_.fit(X_scaled, y_trimmed)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict velocity from lagged features.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins - n_lags, 2)``, float64.
            The first ``n_lags`` bins are consumed by the lag construction.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "WienerFilterDecoder is not fitted. Call fit before predict."
            )
        X_lagged = self._build_lagged_features(X, self.n_lags)
        return self.model_.predict(
            self.scaler_.transform(X_lagged)
        ).astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute R² per velocity dimension.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y : np.ndarray
            Shape ``(n_bins, 2)``. Ground-truth velocity aligned to ``X``.
            The first ``n_lags`` rows are trimmed internally to align with
            the prediction.

        Returns
        -------
        dict[str, float]
            ``{'vx': R²_vx, 'vy': R²_vy}``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        return _r2_dict(y[self.n_lags:], self.predict(X))


# ---------------------------------------------------------------------------
# KalmanFilterDecoder
# ---------------------------------------------------------------------------

class KalmanFilterDecoder:
    """Kalman filter decoder wrapping ``Neural_Decoding.decoders.KalmanFilterDecoder``.

    The Kalman filter is a probabilistic state-space model that maintains a
    belief over the current kinematic state and updates it using both a
    learned transition model (state-to-state) and a learned measurement model
    (state-to-spikes).  Unlike Ridge and Wiener filter, it propagates
    uncertainty across time steps — making it the classical BCI gold standard
    for continuous decoding.

    **Important — predict requires y_test for initialization:**
    The Kalman filter needs an initial state estimate.
    ``Neural_Decoding.KalmanFilterDecoder.predict`` takes ``y_test`` for this
    purpose: ``y_test[0]`` seeds the filter at time 0, and the filter then
    propagates forward using only neural data.  Pass the full test-set
    ground-truth as ``y_test`` — the first row is used for seeding only, not
    as a "cheat" for later time steps.

    Parameters
    ----------
    C : float
        Noise scaling parameter for the transition covariance matrix ``W``.
        Larger ``C`` increases process noise, making the filter trust
        observations more and transitions less.  Default 1.
    lag : int
        Number of bins to shift ``X`` backward relative to ``y`` so that the
        filter uses neural activity from ``lag`` bins ago to predict the
        current kinematic state.  At ``bin_size_ms=50``, ``lag=1`` = 50 ms
        lead time.  Default 0 (no lag).

    Attributes
    ----------
    model_ : Neural_Decoding.decoders.KalmanFilterDecoder
        The fitted underlying decoder.
    _is_fitted : bool
        True after ``fit`` has been called.

    Notes
    -----
    Lag handling: when ``lag > 0``, ``fit`` trains on ``(X[:-lag], y[lag:])``
    and ``predict`` evaluates on ``(X_test[:-lag], y_test[lag:])``.  Both X
    and y are trimmed by ``lag`` rows, so output shapes are
    ``(n_bins - lag, 2)``.

    Examples
    --------
    >>> dec = KalmanFilterDecoder(C=1, lag=0)
    >>> dec.fit(X_train, y_train)
    >>> dec.score(X_test, y_test)
    {'vx': 0.60, 'vy': 0.57}
    """

    def __init__(self, C: float = 1, lag: int = 0) -> None:
        self.C = C
        self.lag = lag
        self.model_ = None
        self._is_fitted: bool = False

    def _apply_lag(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shift X backward relative to y by ``self.lag`` bins.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y : np.ndarray
            Shape ``(n_bins, 2)``.

        Returns
        -------
        X_shifted : np.ndarray
            Shape ``(n_bins - lag, n_features)`` when ``lag > 0``, else ``X``.
        y_aligned : np.ndarray
            Shape ``(n_bins - lag, 2)`` when ``lag > 0``, else ``y``.
        """
        if self.lag > 0:
            return X[: -self.lag], y[self.lag :]
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> KalmanFilterDecoder:
        """Fit the Kalman filter transition and measurement matrices.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``. Binned spike counts.
        y : np.ndarray
            Shape ``(n_bins, 2)``. Velocity targets — column 0 = vx, 1 = vy.

        Returns
        -------
        KalmanFilterDecoder
            self, for method chaining.

        Raises
        ------
        ImportError
            If ``Neural_Decoding`` is not installed.
        """
        from Neural_Decoding.decoders import (
            KalmanFilterDecoder as _KFD,
        )

        X_shifted, y_aligned = self._apply_lag(X, y)
        self.model_ = _KFD(C=self.C)
        self.model_.fit(X_shifted, y_aligned)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Run the Kalman filter forward pass on test data.

        ``y_test[0]`` (after lag trimming) is used solely to initialize the
        filter state at time 0.  The filter then propagates forward using only
        neural observations; it does **not** peek at future ground-truth values.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``. Test-set spike counts.
        y_test : np.ndarray
            Shape ``(n_bins, 2)``. Test-set ground-truth velocity.
            Only the first row (after lag trimming) is used for initialization.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins - lag, 2)``, float64.  Predicted vx and vy at
            each test bin.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "KalmanFilterDecoder is not fitted. Call fit before predict."
            )
        X_shifted, y_aligned = self._apply_lag(X, y_test)
        return np.asarray(
            self.model_.predict(X_shifted, y_aligned), dtype=np.float64
        )

    def score(self, X: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Compute R² per velocity dimension on test data.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y_test : np.ndarray
            Shape ``(n_bins, 2)``. Ground-truth test velocity.

        Returns
        -------
        dict[str, float]
            ``{'vx': R²_vx, 'vy': R²_vy}``.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        y_pred = self.predict(X, y_test)
        _, y_true = self._apply_lag(X, y_test)
        return _r2_dict(y_true, y_pred)
