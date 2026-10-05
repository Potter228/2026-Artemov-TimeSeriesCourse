import numpy as np
import math
import copy
from numba import njit

from modules.utils import sliding_window, z_normalize
from modules.metrics import DTW_distance, _dtw_cost


def apply_exclusion_zone(array: np.ndarray, idx: int, excl_zone: int) -> np.ndarray:
    """Exclude indices from idx-excl_zone through idx+excl_zone, inclusive."""
    if array.ndim != 1 or not 0 <= idx < len(array) or excl_zone < 0:
        raise ValueError("Invalid profile, index, or exclusion zone")
    start = max(0, idx - excl_zone)
    stop = min(len(array), idx + excl_zone + 1)
    array[start:stop] = np.inf
    return array


def topK_match(dist_profile: np.ndarray, excl_zone: int, topK: int = 3,
               max_distance: float = np.inf) -> dict:
    """Greedily select the nearest windows, excluding trivial matches."""
    profile = np.asarray(dist_profile, dtype=float).copy()
    if profile.ndim != 1 or topK < 0 or excl_zone < 0:
        raise ValueError("Expected a 1D profile and nonnegative topK/exclusion zone")
    profile[~np.isfinite(profile)] = np.inf
    results = {'indices': [], 'distances': []}
    for _ in range(topK):
        if profile.size == 0:
            break
        idx = int(np.argmin(profile))
        distance = float(profile[idx])
        if not np.isfinite(distance) or distance > max_distance:
            break
        results['indices'].append(idx)
        results['distances'].append(distance)
        apply_exclusion_zone(profile, idx, excl_zone)
    return results


class BestMatchFinder:
    """Common search parameters: exclusion fraction, topK, normalization, r."""

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3,
                 is_normalize: bool = True, r: float = 0.05) -> None:
        if not 0 <= excl_zone_frac <= 1 or not 0 <= r <= 1 or topK < 1:
            raise ValueError("Invalid search parameters")
        self.excl_zone_frac = excl_zone_frac
        self.topK = topK
        self.is_normalize = is_normalize
        self.r = r

    def _calculate_excl_zone(self, m: int) -> int:
        return math.ceil(m * self.excl_zone_frac)

    def perform(self):
        raise NotImplementedError


@njit
def _naive_dtw_profile(windows: np.ndarray, query: np.ndarray,
                       is_normalize: bool, radius: int) -> np.ndarray:
    """Compute DTW for every candidate without lower-bound pruning."""
    profile = np.full(len(windows), np.inf)
    m = len(query)
    candidate = np.empty(m)
    for i in range(len(windows)):
        mean, std = 0.0, 1.0
        if is_normalize:
            for j in range(m):
                mean += windows[i, j]
            mean /= m
            variance = 0.0
            for j in range(m):
                variance += (windows[i, j] - mean) ** 2
            std = np.sqrt(variance / m)
            if std == 0:
                continue
        for j in range(m):
            candidate[j] = (windows[i, j] - mean) / std
        profile[i] = _dtw_cost(candidate, query, radius)
    return profile


class NaiveBestMatchFinder(BestMatchFinder):
    """Exhaustive DTW subsequence search followed by topK selection."""

    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        ts_data = np.asarray(ts_data, dtype=float)
        query = np.asarray(query, dtype=float)
        if query.ndim != 1 or query.size == 0 or ts_data.ndim not in (1, 2):
            raise ValueError("Expected a nonempty 1D query and a series or window matrix")
        if not (np.isfinite(ts_data).all() and np.isfinite(query).all()):
            raise ValueError("Time series must contain finite values")
        if ts_data.ndim == 1:
            windows = sliding_window(ts_data, len(query))
        else:
            windows = ts_data
        if windows.shape[0] == 0 or windows.shape[1] != len(query):
            raise ValueError("Windows must be nonempty and match the query length")
        if self.is_normalize:
            query = z_normalize(query)
        self.dist_profile_ = _naive_dtw_profile(
            windows, query, self.is_normalize, int(self.r * len(query)))
        # Keep the full profile: early pruning by a changing topK threshold can
        # incorrectly discard windows that become eligible after exclusions.
        return topK_match(self.dist_profile_, self._calculate_excl_zone(len(query)), self.topK)


class UCR_DTW(BestMatchFinder):
    """
    UCR-DTW Match Finder
    
    Additional parameters
    ----------
    not_pruned_num: number of non-pruned subsequences
    lb_Kim_num: number of subsequences that pruned by LB_Kim bounding
    lb_KeoghQC_num: number of subsequences that pruned by LB_KeoghQC bounding
    lb_KeoghCQ_num: number of subsequences that pruned by LB_KeoghCQ bounding
    """

    def __init__(self, excl_zone_frac: float = 1, topK: int = 3, is_normalize: bool = True, r: float = 0.05):
        super().__init__(excl_zone_frac, topK, is_normalize, r)
        """ 
        Constructor of class UCR_DTW
        """        

        self.not_pruned_num = 0
        self.lb_Kim_num = 0
        self.lb_KeoghQC_num = 0
        self.lb_KeoghCQ_num = 0


    def _LB_Kim(self, subs1: np.ndarray, subs2: np.ndarray) -> float:
        """
        Compute LB_Kim lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        
        Returns
        -------
        lb_Kim: LB_Kim lower bound
        """

        lb_Kim = 0
        
        # INSERT YOUR CODE

        return lb_Kim


    def _LB_Keogh(self, subs1: np.ndarray, subs2: np.ndarray, r: float) -> float:
        """
        Compute LB_Keogh lower bound between two subsequences
        
        Parameters
        ----------
        subs1: the first subsequence
        subs2: the second subsequence
        r: warping window size
        
        Returns
        -------
        lb_Keogh: LB_Keogh lower bound
        """

        lb_Keogh = 0

        # INSERT YOUR CODE

        return lb_Keogh


    def get_statistics(self) -> dict:
        """
        Return statistics on the number of pruned and non-pruned subsequences of a time series   
        
        Returns
        -------
            dictionary containing statistics
        """

        statistics = {
            'not_pruned_num': self.not_pruned_num,
            'lb_Kim_num': self.lb_Kim_num,
            'lb_KeoghCQ_num': self.lb_KeoghCQ_num,
            'lb_KeoghQC_num': self.lb_KeoghQC_num
        }

        return statistics


    def perform(self, ts_data: np.ndarray, query: np.ndarray) -> dict:
        """
        Search subsequences in a time series that most closely match the query using UCR-DTW algorithm
        
        Parameters
        ----------
        ts_data: time series
        query: query, shorter than time series

        Returns
        -------
        best_match: dictionary containing results of UCR-DTW algorithm
        """

        query = copy.deepcopy(query)
        if (len(ts_data.shape) != 2): # time series set
            ts_data = sliding_window(ts_data, len(query))

        N, m = ts_data.shape

        excl_zone = self._calculate_excl_zone(m)

        dist_profile = np.ones((N,))*np.inf
        bsf = np.inf
        
        bestmatch = {
            'index' : [],
            'distance' : []
        }

        # INSERT YOUR CODE

        return bestmatch
