import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)
    if ts1.ndim != 1 or ts1.shape != ts2.shape or ts1.size == 0:
        raise ValueError("Expected nonempty one-dimensional series of equal length")
    if not (np.isfinite(ts1).all() and np.isfinite(ts2).all()):
        raise ValueError("Time series must contain finite values")
    ed_dist = float(np.sqrt(np.sum((ts1 - ts2) ** 2)))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)
    if ts1.ndim != 1 or ts1.shape != ts2.shape or ts1.size == 0:
        raise ValueError("Expected nonempty one-dimensional series of equal length")
    if not (np.isfinite(ts1).all() and np.isfinite(ts2).all()):
        raise ValueError("Time series must contain finite values")
    n = len(ts1)
    mean1, mean2 = np.mean(ts1), np.mean(ts2)
    std1, std2 = np.std(ts1), np.std(ts2)
    if std1 == 0 or std2 == 0:
        raise ValueError("Z-normalization is undefined for constant series")
    # Centered dot product is equivalent to <ts1, ts2> - n * mean1 * mean2.
    correlation = np.dot(ts1 - mean1, ts2 - mean2) / (n * std1 * std2)
    correlation = np.clip(correlation, -1.0, 1.0)
    norm_ed_dist = float(np.sqrt(abs(2 * n * (1 - correlation))))
    # Avoid cancellation in 1 - correlation for almost identical shapes.
    if correlation > 1 - 1e-12:
        norm_ed_dist = ED_distance((ts1 - mean1) / std1, (ts2 - mean2) / std2)

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping radius as a fraction of series length, from 0 to 1
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    ts1 = np.asarray(ts1, dtype=float)
    ts2 = np.asarray(ts2, dtype=float)
    if ts1.ndim != 1 or ts1.shape != ts2.shape or ts1.size == 0:
        raise ValueError("Expected nonempty one-dimensional series of equal length")
    if not (np.isfinite(ts1).all() and np.isfinite(ts2).all()):
        raise ValueError("Time series must contain finite values")
    if not 0 <= r <= 1:
        raise ValueError("r must be between 0 and 1")
    n = len(ts1)
    radius = int(r * n)
    costs = np.full((n + 1, n + 1), np.inf)
    costs[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - radius), min(n, i + radius) + 1):
            costs[i, j] = (ts1[i - 1] - ts2[j - 1]) ** 2 + min(
                costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1])
    # No square root: this is the accumulated squared cost in the assignment.
    dtw_dist = float(costs[n, n])

    return dtw_dist
