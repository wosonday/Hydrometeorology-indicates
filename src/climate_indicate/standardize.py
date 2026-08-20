"""
Core for all standardize index (SPI, SSI, Streamflow index,...)

This is the module deepest in the package: Just `standardize()`,
it encapsulates the entire logic for reshaping data by year and fitting separate distributions for each period
(e.g., all January values ​​across different years are fitted to a single distribution, distinct from February, etc.).
This approach avoids mixing seasonal effects and adheres strictly to the standard SPI methodology.
"""

from __future__ import annotations
import numpy as np
from scipy import stats

from climate_indicate.distributions import Distribution

__all__ = ["standardize"]


def _pad_to_multiple(values: np.ndarray, mutiple: int) -> np.ndarray:
    """
    add NaN in the end of array to can divisible by `mutiple`,
    Allows reshape into matrix (year, period/year) without cutting data
    """
