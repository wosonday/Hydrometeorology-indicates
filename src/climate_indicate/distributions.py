"""
Statistic distribution port + spetific adapters

There are four different adapters (Gamma, Log-nomal, Pearson III and Non-parametric KDE)that all satisfy the same interface;
`standardize.py` only interacts with the `Distribution` interface and is unaware of the internal details of each adapter.
"""

from __future__ import annotations

from turtle import backward
from typing import Any, Protocol

import numpy as np
from scipy import stats

__all__ =  []

_MIN_FIT_SAMPLES = 4

class Distribution(Protocol):
    """
    Parent class of distribution
    ---
    standardize.py call by two thís method
    """
    def fit(self, values: np.ndarray) -> Any:
       ...

    def cdf(self, values: np.ndarray, params: Any) -> np.ndarray:
        ...

class GammaDistribution:
    """
    Suitable for precipretation - data shift to the right,
    Zero precipitation is handled separately.
    """

    def fit(self, values: np.ndarray) -> tuple[float, float]:
        if len(values) < _MIN_FIT_SAMPLES:
            raise ValueError(f"Need min {_MIN_FIT_SAMPLES} positive samples to fit Gamma")
        alpha, _loc, beta = stats.gamma.fit(values, floc=0)
        return alpha, beta

    def cdf(self, values: np.ndarray, params: tuple[float, float]) -> np.ndarray:
        alpha, beta = params
        return stats.gamma.cdf(values, a=alpha, scale=beta)


class LogNomalDistribution:
    """
    Suitable for steady streamflow, less extreme
    """

    def fit(self, values: np.ndarray) -> tuple[float, float]:
        if len(values) < _MIN_FIT_SAMPLES:
            raise ValueError(f"Need min {_MIN_FIT_SAMPLES} positive samples to fit Log-normal, have {len(values)}")
        shape, _loc, scale = stats.lognorm.fit(values, floc=0)
        return shape, scale

    def cdf(self, values: np.ndarray, params: tuple[float, float]) -> np.ndarray:
        shape, scale = params
        return stats.lognorm.cdf(values, s=shape, scale=scale)

class Pearson3Distribution:
    """
    Suitable for highly variable flows—using the same distribution family that SPEI and Palmer employ in climatological literature.
    """

    def fit(self, values: np.ndarray) -> tuple[float, float, float]:
        if len(values) < _MIN_FIT_SAMPLES:
            raise ValueError(f"Need min {_MIN_FIT_SAMPLES} positive samples to fit Pearson III, have {len(values)}")
        skew, loc, scale = stats.pearson3.fit(values)
        return skew, loc, scale

    def cdf(self, values: np.ndarray, params: tuple[float, float, float]) -> np.ndarray:
        skew, loc, scale = params
        return stats.pearson3.cdf(values, skew=skew, loc=loc, scale=scale)


class EmpiricalKDEDistribution:
    """
    Non-parametric — estimating probability density using Gaussian kernel
    density estimation (KDE), followed by numerical integration to obtain
    the empirical CDF.

    Does not assume a specific distribution family — suitable for soil moisture,
    which is bounded within the range [0, saturation] and does not always
    follow a clear standard distribution family like precipitation or streamflow.
    """

    def __init__(self, grid_points: int = 2000) -> None:
        self._grid_points = grid_points

    def fit(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(values) < _MIN_FIT_SAMPLES:
            raise ValueError(f"Need min {_MIN_FIT_SAMPLES} to fit KDE, have {len(values)}")
        if np.allclose(values, values[0]):
            raise ValueError("Cannot fit KDE when all values are identical (variance = 0)")

        kde = stats.gaussian_kde(values)
        spread = float(values.std(ddof=1))
        pad = 3 * kde.factor * spread
        grid = np.linspace(values.min() - pad, values.max() + pad, self._grid_points)
        pdf = kde.evaluate(grid)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        return grid, cdf

    def cdf(self, values: np.ndarray, params: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        grid, cdf_grid = params
        return np.clip(np.interp(values, grid, cdf_grid), 0.0, 1.0)
