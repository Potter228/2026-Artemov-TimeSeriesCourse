from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from stumpy import snippets


ACTIVITY_NAMES = {12: 'Подъем по лестнице', 13: 'Спуск по лестнице'}


def load_pamap(path, start=1790.0, stop=1930.0):
    """Read either orientation, then select actual timestamps inclusively."""
    raw = np.loadtxt(Path(path))
    if raw.ndim != 2:
        raise ValueError('Expected a two-dimensional PAMAP table.')
    if raw.shape[0] == 20 and raw.shape[1] != 20:
        raw = raw.T
    if raw.shape[1] != 20:
        raise ValueError('Expected 20 fields described in the README.')
    if not np.isfinite(raw[:, 0]).all() or not np.all(np.diff(raw[:, 0]) > 0):
        raise ValueError('Timestamps must be finite and strictly increasing.')
    selected = raw[(raw[:, 0] >= start) & (raw[:, 0] <= stop)]
    if selected.size == 0:
        raise ValueError('No samples in the selected time interval.')
    columns = ['time', 'activity', 'gyro_X', 'gyro_Y', 'gyro_Z']
    frame = pd.DataFrame(selected[:, [0, 1, 10, 11, 12]], columns=columns)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Selected timestamps, labels or gyroscope values contain missing data.')
    if not np.all(frame['activity'] == frame['activity'].astype(int)):
        raise ValueError('Activity IDs must be integers.')
    frame['activity'] = frame['activity'].astype(int)
    return frame


def snippet_labels(result):
    """Decode half-open STUMPY regimes at window STARTS, without padding."""
    profiles = np.asarray(result[2], dtype=float)
    if profiles.ndim != 2 or not np.isfinite(profiles).all():
        raise ValueError('Expected finite snippet distance profiles.')
    labels = np.full(profiles.shape[1], -1, dtype=int)
    for idx, start, stop in np.asarray(result[5], dtype=int):
        if not (0 <= idx < len(profiles) and 0 <= start < stop <= len(labels)):
            raise ValueError('Invalid snippet regime.')
        if np.any(labels[start:stop] != -1):
            raise ValueError('Overlapping snippet regimes.')
        labels[start:stop] = idx
    if np.any(labels < 0):
        raise ValueError('Snippet regimes do not cover all valid windows.')
    np.testing.assert_array_equal(labels, np.argmin(profiles, axis=0))
    return labels


def align_to_reference(labels, reference, k=2):
    """Permute cluster IDs using prediction overlap only, without activity labels."""
    labels, reference = np.asarray(labels, dtype=int), np.asarray(reference, dtype=int)
    if labels.shape != reference.shape or labels.ndim != 1:
        raise ValueError('Both label vectors must have the same one-dimensional shape.')
    if np.any((labels < 0) | (labels >= k) | (reference < 0) | (reference >= k)):
        raise ValueError('Cluster IDs must lie in [0, k).')
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (labels, reference), 1)
    row, col = linear_sum_assignment(-counts)
    mapping = dict(zip(row.tolist(), col.tolist()))
    return np.array([mapping[int(value)] for value in labels]), mapping


def activity_mapping(labels, truth, k=2):
    """Match anonymous clusters to activity IDs for descriptive clustering accuracy."""
    labels, truth = np.asarray(labels, dtype=int), np.asarray(truth, dtype=int)
    classes = np.unique(truth)
    if labels.shape != truth.shape or len(classes) != k:
        raise ValueError('This one-to-one evaluation requires k observed activity classes.')
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (labels, np.searchsorted(classes, truth)), 1)
    rows, cols = linear_sum_assignment(-counts)
    return {int(row): int(classes[col]) for row, col in zip(rows, cols)}


def decode_activities(labels, mapping):
    return np.array([mapping[int(value)] for value in labels], dtype=int)


def majority_vote(label_vectors, reference_index=1):
    """Vote across aligned axes. In a tie prefer the reference axis, then lowest ID."""
    values = np.asarray(label_vectors, dtype=int)
    if values.ndim != 2 or values.shape[0] < 1 or np.any(values < 0):
        raise ValueError('Expected nonnegative aligned cluster labels, axes by windows.')
    if not 0 <= reference_index < len(values):
        raise ValueError('Invalid reference axis.')
    ids = np.unique(values)
    counts = np.stack([(values == idx).sum(axis=0) for idx in ids])
    best = counts.max(axis=0)
    vote = ids[np.argmax(counts, axis=0)]
    reference = values[reference_index]
    reference_counts = (values == reference).sum(axis=0)
    tied = (counts == best).sum(axis=0) > 1
    vote[tied & (reference_counts == best)] = reference[tied & (reference_counts == best)]
    return vote


def snippet_table(result, timestamps):
    m = result[0].shape[1]
    return pd.DataFrame({
        'Сниппет': np.arange(len(result[1])),
        'Начало окна': result[1],
        'Начало, с': np.asarray(timestamps)[result[1]],
        'Конец, с': np.asarray(timestamps)[result[1] + m - 1],
        'Доля окон': result[3],
    })


def parameter_sweep(signals, m, max_k=9, percentage=0.5):
    """One public STUMPY call per axis; greedy prefixes give results for every k."""
    results, areas = {}, {}
    for name, values in signals.items():
        result = snippets(np.asarray(values, dtype=float), m, max_k, percentage=percentage)
        # Area is the integral (discrete sum) of the pointwise minimum of selected profiles.
        verified = np.minimum.accumulate(result[2], axis=0).sum(axis=1)
        np.testing.assert_allclose(result[4], verified, rtol=1e-10, atol=1e-6)
        if not np.all(np.isfinite(verified)) or np.any(verified <= 0):
            raise ValueError('Profile areas must be finite and positive for this ratio.')
        results[name], areas[name] = result, verified
    area_table = pd.DataFrame(areas, index=pd.Index(range(1, max_k+1), name='k'))
    area_table['Среднее XYZ'] = area_table.mean(axis=1)
    change_table = area_table.shift(1) / area_table - 1
    assert np.all(area_table.diff().iloc[1:].to_numpy() <= 1e-6)
    preferred_k = int(change_table['Среднее XYZ'].idxmax())
    return area_table, change_table, preferred_k, results
