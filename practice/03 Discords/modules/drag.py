"""Two-pass DRAG with exact z-normalized Euclidean distances.

Numba compiles the scans; only candidates surviving both directional passes
need an exact nearest-neighbor distance during refinement. The default
exclusion radius is STUMPY's ceil(m / 4), as in the supplied lab template.
"""
import numpy as np
from numba import njit
from stumpy import core, config


def _get_chunks_ranges(a, shift=None):
    a = np.asarray(a, dtype=int)
    if not len(a):
        return np.empty((0, 2), dtype=int)
    cuts = np.flatnonzero(np.diff(a) != 1) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts - 1, len(a) - 1]
    return np.column_stack((a[starts], a[stops] + 1 + (shift or 0)))


def _windows(T, m, M_T, sigma):
    windows = np.lib.stride_tricks.sliding_window_view(T, m)
    constant = np.asarray(sigma) == 0
    scale = np.where(constant, 1.0, sigma)
    normalized = (windows - np.asarray(M_T)[:, None]) / scale[:, None]
    return np.ascontiguousarray(normalized), constant


@njit(cache=True)
def _squared_distance(windows, constant, i, j, bound):
    if constant[i] and constant[j]:
        return 0.0
    if constant[i] or constant[j]:
        return float(windows.shape[1])
    distance = 0.0
    for k in range(windows.shape[1]):
        distance += (windows[i, k] - windows[j, k]) ** 2
        if distance > bound:
            break
    return distance


@njit(cache=True)
def _directional_scan(windows, constant, candidates, radius, threshold, right):
    size = len(windows)
    for i in range(size):
        if not candidates[i]:
            continue
        start, stop = (i + radius + 1, size) if right else (0, max(0, i - radius))
        for j in range(start, stop):
            if _squared_distance(windows, constant, i, j, threshold) < threshold:
                candidates[i] = False
                break
    return candidates


def find_candidates(T, m, M_T, Σ_T, r, init_cands=None, right=True, finite=False):
    """Keep windows with no nontrivial right (left) neighbor closer than r.

    A candidate without a neighbor on this side survives until the other pass.
    init_cands restricts possible discords, never their comparison neighbors.
    """
    T = np.asarray(T, dtype=float)
    if not np.isfinite(r) or r < 0:
        raise ValueError('r must be finite and nonnegative')
    k = len(T) - m + 1
    candidates = np.ones(k, dtype=bool) if init_cands is None else np.array(init_cands, dtype=bool, copy=True)
    if candidates.shape != (k,):
        raise ValueError('Candidate mask has an invalid shape')
    candidates &= np.isfinite(M_T) & np.isfinite(Σ_T)
    windows, constant = _windows(T, m, M_T, Σ_T)
    radius = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    return _directional_scan(windows, constant, candidates, radius, float(r) ** 2, bool(right))


@njit(cache=True)
def _nearest_candidates(windows, constant, candidates, radius):
    profile = np.full(len(windows), -np.inf)
    neighbors = np.full(len(windows), -1, dtype=np.int64)
    for i in range(len(windows)):
        if not candidates[i]:
            continue
        nearest = np.inf
        for j in range(len(windows)):
            if abs(i - j) <= radius:
                continue
            distance = _squared_distance(windows, constant, i, j, nearest)
            if distance < nearest:
                nearest = distance
                neighbors[i] = j
        if np.isfinite(nearest):
            profile[i] = np.sqrt(nearest)
    return profile, neighbors


def refine_candidates(T, m, M_T, Σ_T, is_cands):
    """Rank surviving candidates and suppress nearby copies of each discord."""
    windows, constant = _windows(np.asarray(T, dtype=float), m, M_T, Σ_T)
    candidates = np.asarray(is_cands, dtype=bool)
    if candidates.shape != (len(windows),):
        raise ValueError('Candidate mask has an invalid shape')
    radius = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    profile, neighbors = _nearest_candidates(windows, constant, candidates, radius)
    indices, distances, nn_indices = [], [], []
    while np.isfinite(profile).any():
        idx = int(np.argmax(profile))
        indices.append(idx)
        distances.append(float(profile[idx]))
        nn_indices.append(int(neighbors[idx]))
        profile[max(0, idx - radius):idx + radius + 1] = -np.inf
    return indices, distances, nn_indices


def DRAG(data, m, r, include=None):
    """Gather on the right, clean on the left, then refine range discords."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError('Expected a finite one-dimensional series')
    if not isinstance(m, (int, np.integer)) or not 3 <= m <= len(data):
        raise ValueError('Invalid window length')
    if not np.isfinite(r) or r < 0:
        raise ValueError('r must be finite and nonnegative')
    if include is not None:
        include = np.asarray(include, dtype=bool)[:len(data) - m + 1].copy()
        if include.shape != (len(data) - m + 1,):
            raise ValueError('include has an invalid shape')
    T, M_T, Σ_T, *_ = core.preprocess(data, m)
    candidates = find_candidates(T, m, M_T, Σ_T, r, include, right=True)
    candidates = find_candidates(T, m, M_T, Σ_T, r, candidates, right=False)
    return refine_candidates(T, m, M_T, Σ_T, candidates)
