import numpy as np
import stumpy
from stumpy import config


def compute_mp(ts1: np.ndarray, m: int, exclusion_zone: int = None,
               ts2: np.ndarray = None) -> dict:
    """Compute an exact z-normalized self-join profile using STUMPY.

    The exclusion zone is an inclusive radius in window-start indices.
    Only the self-join required in parts 1 and 2 is implemented here.
    """
    if ts2 is not None:
        raise NotImplementedError('The AB-join belongs to part 3.')
    values = np.asarray(ts1, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('ts1 must be a finite one-dimensional time series.')
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or not 3 <= m <= len(values):
        raise ValueError('m must be an integer between 3 and len(ts1).')
    m = int(m)
    if exclusion_zone is None:
        exclusion_zone = int(np.ceil(m / 4))
    if isinstance(exclusion_zone, bool) or not isinstance(exclusion_zone, (int, np.integer)) or exclusion_zone < 0:
        raise ValueError('exclusion_zone must be a nonnegative integer.')
    exclusion_zone = int(exclusion_zone)
    if len(values) - m <= exclusion_zone:
        raise ValueError('There are no windows outside the exclusion zone.')

    # STUMPY uses ceil(m / denominator). Restore the setting after this call.
    previous = config.STUMPY_EXCL_ZONE_DENOM
    denominator = (np.inf if exclusion_zone == 0 else
                   np.nextafter(m / exclusion_zone, np.inf))
    assert int(np.ceil(m / denominator)) == exclusion_zone
    try:
        config.STUMPY_EXCL_ZONE_DENOM = denominator
        profile = stumpy.stump(values, m, ignore_trivial=True)
    finally:
        config.STUMPY_EXCL_ZONE_DENOM = previous
    return {
        'mp': np.asarray(profile[:, 0], dtype=np.float64),
        'mpi': np.asarray(profile[:, 1], dtype=np.int64),
        'm': m,
        'excl_zone': exclusion_zone,
        'data': {'ts1': values.copy(), 'ts2': None},
    }
