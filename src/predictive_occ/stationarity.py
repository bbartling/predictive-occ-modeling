"""
Stationarity Tests
==================

This module wraps common statistical tests for determining whether
a time series is stationary.  It is a direct refactor of the
``src/stationarity`` module and exposes the same public functions
``adf_stationary`` and ``kpss_stationary``.

The Augmented Dickey–Fuller (ADF) test checks for the presence of a
unit root (non‑stationarity), while the Kwiatkowski–Phillips–Schmidt–Shin
(KPSS) test uses stationarity as its null hypothesis.  Both tests
return a boolean indicating whether the series appears stationary at
the given significance level along with the p‑value and the full
result tuple returned by statsmodels.
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
    is_stat = p_value > alpha
    return is_stat, p_value, (stat, p_value, lags, critical_values)


__all__ = ["adf_stationary", "kpss_stationary"]