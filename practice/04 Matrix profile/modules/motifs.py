import numpy as np
from modules.utils import apply_exclusion_zone


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """Greedily select the lowest-distance available nearest-neighbor pairs.

    Both members of each pair and their exclusion zones are suppressed.
    A later pair cannot reuse either a suppressed start or neighbor.
    The input matrix profile is not changed.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer.')
    distances = np.asarray(matrix_profile['mp'], dtype=float).copy()
    neighbors = np.asarray(matrix_profile['mpi'], dtype=int)
    radius = int(matrix_profile['excl_zone'])
    available = np.isfinite(distances) & (neighbors >= 0) & (neighbors < len(distances))
    motifs_idx, motifs_dist = [], []
    for _ in range(int(top_k)):
        valid = available.copy()
        starts = np.flatnonzero(valid)
        valid[starts] &= available[neighbors[starts]]
        valid &= np.abs(np.arange(len(distances)) - neighbors) > radius
        if not valid.any():
            break
        candidate = int(np.argmin(np.where(valid, distances, np.inf)))
        neighbor = int(neighbors[candidate])
        motifs_idx.append(tuple(sorted((candidate, neighbor))))
        motifs_dist.append(float(distances[candidate]))
        for idx in (candidate, neighbor):
            apply_exclusion_zone(available, idx, radius, False)
            apply_exclusion_zone(distances, idx, radius, np.inf)
    return {'indices': motifs_idx, 'distances': motifs_dist}
