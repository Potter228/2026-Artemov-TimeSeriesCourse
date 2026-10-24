import numpy as np


def is_nan_inf(arr: np.ndarray) -> bool:
    """Return whether a scalar or an array contains a nonfinite value."""
    return bool(np.any(~np.isfinite(arr)))


def apply_exclusion_zone(a: np.ndarray, idx: int, excl_zone: int,
                         val: float) -> np.ndarray:
    """Set an inclusive, symmetric exclusion zone in place."""
    start = max(0, int(idx) - int(excl_zone))
    stop = min(a.shape[-1], int(idx) + int(excl_zone) + 1)
    a[..., start:stop] = val
    return a
