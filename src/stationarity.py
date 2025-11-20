"""
Stationarity Tests
==================

This module wraps common statistical tests for determining whether
a time series is stationary.  Stationarity means that the
statistical properties of the series—mean, variance and
autocorrelation—do not change over time.  Many forecasting
techniques, including ARIMA and linear regression models, assume
stationarity in the input data.  When the data are strongly
stationary, simple probability models can be very effective and
interpretable.

Two complementary tests are provided:

``adf_stationary``
    Performs the Augmented Dickey–Fuller (ADF) test.  The ADF test
    examines the presence of a unit root.  The null hypothesis is
    that the series is non‑stationary, so a small p‑value (e.g.
    < 0.05) allows us to reject non‑stationarity and conclude the
    series is stationary【502360774470064†L199-L212】.

``kpss_stationary``
    Performs the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test.  In
    contrast to ADF, the null hypothesis for the KPSS test is that
    the series is stationary.  A large p‑value (e.g. > 0.05)
    indicates stationarity, while a small p‑value suggests
    non‑stationarity【502360774470064†L243-L253】.

In addition to returning a boolean decision, these functions also
return the p‑value and the full test result tuple returned by
statsmodels.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def adf_stationary(
    series: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[bool, float, Tuple]:
    """Perform the Augmented Dickey–Fuller (ADF) test for stationarity.

    Parameters
    ----------
    series : array‑like
        The numeric time series to test.  Missing values will be
        dropped automatically by the underlying statsmodels function.
    alpha : float, default ``0.05``
        Significance level.  If the p‑value returned by the test is
        less than ``alpha`` then the null hypothesis of non‑stationarity
        is rejected and the function returns ``True``.

    Returns
    -------
    is_stationary : bool
        ``True`` if the series is stationary at the specified
        significance level, ``False`` otherwise.
    p_value : float
        The p‑value returned by the test.
    result : tuple
        Full result object returned by :func:`statsmodels.tsa.stattools.adfuller`.
    """
    # Drop NaNs to avoid errors in the adfuller function
    y = np.asarray(series).astype(float)
    y = y[~np.isnan(y)]
    test_stat, p_value, usedlag, nobs, critical_values, icbest = adfuller(
        y, autolag="AIC"
    )
    is_stat = p_value < alpha
    return is_stat, p_value, (test_stat, p_value, usedlag, nobs, critical_values, icbest)


def kpss_stationary(
    series: np.ndarray,
    alpha: float = 0.05,
    regression: str = "c",
) -> Tuple[bool, float, Tuple]:
    """Perform the KPSS test for stationarity.

    Parameters
    ----------
    series : array‑like
        The numeric time series to test.  Missing values will be
        dropped automatically.
    alpha : float, default ``0.05``
        Significance level.  For KPSS the null hypothesis is
        stationarity, so if the p‑value is greater than ``alpha`` the
        series is considered stationary.
    regression : {"c", "ct"}, default ``"c"``
        Whether to include a constant (``"c"``) or constant and trend
        (``"ct"``) in the test.  For detecting stationarity around a
        deterministic trend use ``"ct"``.

    Returns
    -------
    is_stationary : bool
        ``True`` if the series is stationary at the specified
        significance level, ``False`` otherwise.
    p_value : float
        The p‑value returned by the test.
    result : tuple
        Full result object returned by :func:`statsmodels.tsa.stattools.kpss`.
    """
    y = np.asarray(series).astype(float)
    y = y[~np.isnan(y)]
    stat, p_value, lags, critical_values = kpss(y, regression=regression, nlags="auto")
    # For KPSS the null hypothesis is stationarity, so we flip the
    # comparison relative to ADF.
    is_stat = p_value > alpha
    return is_stat, p_value, (stat, p_value, lags, critical_values)