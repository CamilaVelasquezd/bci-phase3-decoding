from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


class RidgeDecoder:
    """Regularized linear decoder for continuous kinematic targets.

    Wraps :class:`sklearn.linear_model.Ridge` with an internal
    :class:`~sklearn.preprocessing.StandardScaler` and a multi-output
    ``score`` that returns R² per output dimension separately.

    Parameters
    ----------
    alpha : float
        L2 regularization strength passed directly to ``Ridge``. Default 1.0.

    Attributes
    ----------
    model_ : Ridge
        Fitted Ridge regression model.
    scaler_ : StandardScaler
        Fitted scaler (zero mean, unit variance).
    _is_fitted : bool
        True after ``fit`` has been called.

    Examples
    --------
    >>> dec = RidgeDecoder(alpha=1.0)
    >>> dec.fit(X_train, y_train)
    >>> r2 = dec.score(X_test, y_test)   # shape (2,) — R² for vx and vy
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
            Shape ``(n_bins, n_outputs)``. Continuous kinematic targets,
            e.g. velocity ``(vx, vy)`` aligned to the same bins as ``X``.

        Returns
        -------
        RidgeDecoder
            The fitted decoder (self), for chaining.
        """
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X_scaled, y)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict kinematic targets for new data.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``. Must have the same feature layout
            as the data passed to ``fit``.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins, n_outputs)``, float64.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "RidgeDecoder is not fitted. Call fit before predict."
            )
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled).astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute R² score per output dimension.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y : np.ndarray
            Shape ``(n_bins, n_outputs)``. Ground-truth targets.

        Returns
        -------
        np.ndarray
            Shape ``(n_outputs,)``. R² for each output dimension separately
            (e.g. ``[R²_vx, R²_vy]``).

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        y_pred = self.predict(X)
        return r2_score(y, y_pred, multioutput="raw_values")


class WienerFilterDecoder:
    """Wiener filter: linear decoder with time-lagged spike bins.

    Builds a lagged feature matrix from ``X`` before fitting, so the decoder
    can exploit neural activity from the previous ``n_lags`` bins when
    predicting the kinematic target at the current bin.  This is the classical
    BCI Wiener filter baseline.

    The lag matrix has ``n_bins - n_lags`` rows; both X and y are trimmed
    accordingly so that the first decodable bin is at index ``n_lags``.

    Parameters
    ----------
    n_lags : int
        Number of past time bins to include as additional features. Default 5.
        With ``bin_size_ms=50`` this corresponds to 250 ms of history.

    Attributes
    ----------
    model_ : LinearRegression
        Fitted ordinary least-squares regression model.
    scaler_ : StandardScaler
        Fitted scaler applied to the lagged feature matrix.
    _is_fitted : bool
        True after ``fit`` has been called.

    Notes
    -----
    ``_build_lagged_features(X, n_lags)`` stacks the current bin alongside
    ``n_lags`` previous bins column-wise, yielding a matrix of shape
    ``(n_bins - n_lags, n_features * (n_lags + 1))``.

    Examples
    --------
    >>> dec = WienerFilterDecoder(n_lags=5)
    >>> dec.fit(X_train, y_train)
    >>> r2 = dec.score(X_test, y_test)   # shape (2,) — R² for vx and vy
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
            Number of past bins to append as additional features.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins - n_lags, n_features * (n_lags + 1))``, float64.
            Row ``i`` concatenates ``X[i + n_lags]`` (lag 0) through
            ``X[i]`` (lag ``n_lags``), so the first ``n_lags`` rows of the
            original ``X`` are consumed and not represented in the output.
        """
        n_bins, n_features = X.shape
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
            Shape ``(n_bins, n_outputs)``. Targets aligned to X bins.
            The first ``n_lags`` rows are trimmed internally to align with
            the lag matrix.

        Returns
        -------
        WienerFilterDecoder
            The fitted decoder (self), for chaining.
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
        """Predict kinematic targets from lagged features.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins - n_lags, n_outputs)``, float64.
            The first ``n_lags`` bins of ``X`` are consumed by lag construction
            and have no corresponding prediction.

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
        X_scaled = self.scaler_.transform(X_lagged)
        return self.model_.predict(X_scaled).astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute R² score per output dimension.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y : np.ndarray
            Shape ``(n_bins, n_outputs)``. Ground-truth targets aligned to X.
            The first ``n_lags`` rows are trimmed internally to match the lag
            matrix.

        Returns
        -------
        np.ndarray
            Shape ``(n_outputs,)``. R² for each output separately
            (e.g. ``[R²_vx, R²_vy]``).

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        y_pred = self.predict(X)
        y_trimmed = y[self.n_lags:]
        return r2_score(y_trimmed, y_pred, multioutput="raw_values")


class WienerRidgeDecoder(WienerFilterDecoder):
    """Wiener-Ridge decoder: time-lagged features with L2 regularization.

    Combines the Wiener filter's temporal lag representation with Ridge
    regression's L2 regularization.  Identical to
    :class:`WienerFilterDecoder` except that the internal OLS model is
    replaced by :class:`sklearn.linear_model.Ridge`, which shrinks
    coefficients toward zero and can improve generalisation when the
    lagged feature matrix is high-dimensional or multicollinear.

    The lag matrix has ``n_bins - n_lags`` rows; both X and y are trimmed
    accordingly so that the first decodable bin is at index ``n_lags``.

    Parameters
    ----------
    n_lags : int
        Number of past time bins to include as additional features. Default 5.
        With ``bin_size_ms=50`` this corresponds to 250 ms of history.
    alpha : float
        L2 regularization strength passed directly to ``Ridge``. Default 1.0.
        Larger values increase shrinkage; ``alpha`` approaching 0 recovers OLS
        (use :class:`WienerFilterDecoder` for pure OLS).

    Attributes
    ----------
    model_ : Ridge
        Fitted Ridge regression model.
    scaler_ : StandardScaler
        Fitted scaler applied to the lagged feature matrix.
    _is_fitted : bool
        True after ``fit`` has been called.

    Notes
    -----
    ``_build_lagged_features``, ``predict``, and ``score`` are inherited
    from :class:`WienerFilterDecoder` unchanged.

    Examples
    --------
    >>> dec = WienerRidgeDecoder(n_lags=5, alpha=1.0)
    >>> dec.fit(X_train, y_train)
    >>> r2 = dec.score(X_test, y_test)   # shape (2,) — R² for vx and vy
    """

    def __init__(self, n_lags: int = 5, alpha: float = 1.0) -> None:
        super().__init__(n_lags=n_lags)
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> WienerRidgeDecoder:
        """Build lag matrix, fit scaler and Ridge regression.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y : np.ndarray
            Shape ``(n_bins, n_outputs)``. Targets aligned to X bins.
            The first ``n_lags`` rows are trimmed internally to align with
            the lag matrix.

        Returns
        -------
        WienerRidgeDecoder
            The fitted decoder (self), for chaining.
        """
        X_lagged = self._build_lagged_features(X, self.n_lags)
        y_trimmed = y[self.n_lags:]
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_lagged)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X_scaled, y_trimmed)
        self._is_fitted = True
        return self


class KalmanFilterDecoder:
    """Kalman filter decoder for continuous kinematic estimation.

    Wraps ``Neural_Decoding.decoders.KalmanFilterDecoder`` with a consistent
    sklearn-style interface.  Unlike Ridge and Wiener filter, the Kalman filter
    propagates uncertainty across time steps using learned transition (A, W) and
    observation (H, Q) matrices.

    ``predict`` requires a ``y_test`` argument: ``y_test[0]`` seeds the initial
    state estimate; the filter then propagates forward using only neural
    observations.

    Parameters
    ----------
    C : float
        Noise scaling constant applied to the transition covariance W.
        Larger values make the filter trust observations more and transitions
        less. Default 1.
    lag : int
        Neural lag in bins. When ``lag > 0`` the decoder fits on
        ``(X[:-lag], y[lag:])`` so that neural activity from ``lag`` bins
        ago predicts the current kinematic state. Default 0.

    Attributes
    ----------
    model_ : Neural_Decoding.decoders.KalmanFilterDecoder
        Fitted underlying decoder.
    _is_fitted : bool
        True after ``fit`` has been called.

    Examples
    --------
    >>> dec = KalmanFilterDecoder(C=1, lag=0)
    >>> dec.fit(X_train, y_train)
    >>> r2 = dec.score(X_test, y_test)   # shape (2,) — R² for vx and vy
    """

    def __init__(self, C: float = 1, lag: int = 0) -> None:
        self.C = C
        self.lag = lag
        self.model_ = None
        self._is_fitted: bool = False

    def _apply_lag(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trim X and y to apply the neural lag.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y : np.ndarray
            Shape ``(n_bins, n_outputs)``.

        Returns
        -------
        X_shifted, y_aligned : np.ndarray
            Both shape ``(n_bins - lag, ...)``. Unchanged when ``lag == 0``.
        """
        if self.lag > 0:
            return X[: -self.lag], y[self.lag :]
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> KalmanFilterDecoder:
        """Estimate transition and observation matrices from training data.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``. Binned spike counts.
        y : np.ndarray
            Shape ``(n_bins, n_outputs)``. Velocity targets aligned to ``X``.

        Returns
        -------
        KalmanFilterDecoder
            self, for method chaining.

        Raises
        ------
        ImportError
            If ``Neural_Decoding`` is not installed (``pip install Neural-Decoding``).
        """
        from Neural_Decoding.decoders import KalmanFilterDecoder as _KFD  # type: ignore[import]

        X_shifted, y_aligned = self._apply_lag(X, y)
        self.model_ = _KFD(C=self.C)
        self.model_.fit(X_shifted, y_aligned)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Run the Kalman filter forward pass on test observations.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``. Test-set spike counts.
        y_test : np.ndarray
            Shape ``(n_bins, n_outputs)``. Ground-truth test targets.
            Only ``y_test[0]`` (after lag trimming) is used to seed the
            initial state; subsequent steps use only neural observations.

        Returns
        -------
        np.ndarray
            Shape ``(n_bins - lag, n_outputs)``, float64.

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

    def score(self, X: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Compute R² score per output dimension.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(n_bins, n_features)``.
        y_test : np.ndarray
            Shape ``(n_bins, n_outputs)``. Ground-truth test targets.

        Returns
        -------
        np.ndarray
            Shape ``(n_outputs,)``. R² for each output dimension separately.

        Raises
        ------
        RuntimeError
            If called before ``fit``.
        """
        y_pred = self.predict(X, y_test)
        _, y_true = self._apply_lag(X, y_test)
        return r2_score(y_true, y_pred, multioutput="raw_values")
