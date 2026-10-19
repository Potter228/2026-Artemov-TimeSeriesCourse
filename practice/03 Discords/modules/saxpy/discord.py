"""Discord discovery routines."""
import numpy as np
from numba import njit
from .visit_registry import VisitRegistry
from .distance import early_abandoned_euclidean
from .znorm import znorm


@njit(cache=True)
def _brute_force_profile(windows, win_size):
    """Exact NN search; all non-overlapping pairs are considered, without SAX."""
    n_windows, m = windows.shape
    profile = np.full(n_windows, np.inf)
    for i in range(n_windows):
        nearest_squared = np.inf
        for j in range(n_windows):
            if abs(i - j) < win_size:
                continue
            squared = 0.0
            for k in range(m):
                squared += (windows[i, k] - windows[j, k]) ** 2
                if squared > nearest_squared:
                    break
            if squared < nearest_squared:
                nearest_squared = squared
        profile[i] = np.sqrt(nearest_squared)
    return profile


def find_discords_brute_force(series, win_size, num_discords=2, znorm_threshold=0.01):
    """Exhaustive search with early-abandoned ED, compiled with Numba.

    The exact NN profile is computed once and reused for topK selection.
    Selected discord windows and each window's NN must not overlap.
    JIT compilation should be warmed up before performance measurements.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1 or not np.isfinite(series).all():
        raise ValueError('Expected a finite one-dimensional series')
    if not isinstance(win_size, (int, np.integer)) or not 1 <= win_size <= len(series):
        raise ValueError('Invalid window length')
    if not isinstance(num_discords, (int, np.integer)) or num_discords < 0:
        raise ValueError('num_discords must be a nonnegative integer')
    windows = np.array([znorm(series[i:i + win_size], znorm_threshold)
                        for i in range(len(series) - win_size + 1)])
    profile = _brute_force_profile(windows, win_size)
    profile[~np.isfinite(profile)] = -np.inf
    discords = []
    for _ in range(num_discords):
        i = int(np.argmax(profile))
        if not np.isfinite(profile[i]):
            break
        discords.append((i, float(profile[i])))
        profile[max(0, i - win_size + 1):i + win_size] = -np.inf
    return discords


def find_best_discord_brute_force(series, win_size, global_registry, znorms):
    """Early-abandoned distance-based discord discovery."""
    best_so_far_distance = -1.0
    best_so_far_index = -1

    outer_registry = global_registry.clone()

    outer_idx = outer_registry.get_next_unvisited()

    while ~np.isnan(outer_idx):

        outer_registry.mark_visited(outer_idx)

        candidate_seq = znorms[outer_idx]

        nn_distance = np.inf
        inner_registry = VisitRegistry(len(series) - win_size + 1)

        inner_idx = inner_registry.get_next_unvisited()

        while ~np.isnan(inner_idx):
            inner_registry.mark_visited(inner_idx)

            if abs(inner_idx - outer_idx) >= win_size:

                curr_seq = znorms[inner_idx]

                dist = early_abandoned_euclidean(candidate_seq, curr_seq, nn_distance)

                if (~np.isnan(dist)) and (dist < nn_distance):
                    nn_distance = dist

            inner_idx = inner_registry.get_next_unvisited()

        if ~(np.inf == nn_distance) and (nn_distance > best_so_far_distance):
            best_so_far_distance = nn_distance
            best_so_far_index = outer_idx

        outer_idx = outer_registry.get_next_unvisited()

    return best_so_far_index, best_so_far_distance
