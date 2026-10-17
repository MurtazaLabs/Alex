"""Shared private helpers for indicator implementations."""

from __future__ import annotations

import numpy as np


def as_float_ndarray(values) -> np.ndarray:
    """Coerce a sequence into a 1-D float64 numpy array."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D series, got {array.ndim} dimension(s)")
    return array


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division that yields NaN where the denominator is zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(numerator, denominator)
        result = np.where(np.isinf(result), np.nan, result)
        result = np.where(result == 0.0, np.nan, result)
    return result
