import numpy as np
from modules.utils import apply_exclusion_zone


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """Select profile maxima, suppressing neighboring window starts."""
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer.')
    distances = np.asarray(matrix_profile['mp'], dtype=float).copy()
    neighbors = np.asarray(matrix_profile['mpi'], dtype=int)
    valid = np.isfinite(distances) & (neighbors >= 0) & (neighbors < len(distances))
    distances[~valid] = -np.inf
    indices, scores, nearest = [], [], []
    for _ in range(int(top_k)):
        idx = int(np.argmax(distances))
        if not np.isfinite(distances[idx]):
            break
        indices.append(idx)
        scores.append(float(distances[idx]))
        nearest.append(int(neighbors[idx]))
        apply_exclusion_zone(distances, idx, matrix_profile['excl_zone'], -np.inf)
    return {'indices': indices, 'distances': scores, 'nn_indices': nearest}
