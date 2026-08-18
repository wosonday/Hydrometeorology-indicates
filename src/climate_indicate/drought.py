from __future__ import annotations

import numpy as np
from scipy import stats


__all__ = ["spi", "ssi", "streamflow_index"]


def _accumulate(values: np.ndarray, scale: int) -> np.ndarray:
    """Tổng trượt `scale` giá trị, đệm NaN đầu để giữ nguyên độ dài
    (không lệch chỉ mục thời gian)."""
    if scale == 1:
        return values.astype(float)
    window_sums = np.convolve(values, np.ones(scale), mode="valid")
    return np.concatenate([np.full(scale - 1, np.nan), window_sums])


def _fit_gamma(nonzero_values: np.ndarray) -> tuple[float, float]:
    alpha, _loc, beta = stats.gamma.fit(nonzero_values, floc=0)
    return alpha, beta


def _to_normal_quantile(accum, q_zero, alpha, beta) -> np.ndarray:
    finite = np.isfinite(accum)
    zeros = (accum == 0) & finite
    cdf = np.full_like(accum, np.nan, dtype=float)
    cdf[finite] = stats.gamma.cdf(accum[finite], a=alpha, scale=beta)
    h = np.clip(np.where(zeros, q_zero, q_zero + (1 - q_zero) * cdf), 1e-6, 1 - 1e-6)
    result = np.full_like(accum, np.nan, dtype=float)
    result[finite] = stats.norm.ppf(h[finite])
    return result


def spi(precip: np.ndarray, scale: int = 3) -> np.ndarray:
    """Tính SPI. Trả về mảng CÙNG độ dài với `precip`."""
    accum = _accumulate(np.asarray(precip, dtype=float), scale)
    finite = np.isfinite(accum)
    if not finite.any():
        return accum
    zeros = (accum == 0) & finite
    q_zero = zeros.sum() / finite.sum()
    alpha, beta = _fit_gamma(accum[finite & ~zeros])
    return _to_normal_quantile(accum, q_zero, alpha, beta)
