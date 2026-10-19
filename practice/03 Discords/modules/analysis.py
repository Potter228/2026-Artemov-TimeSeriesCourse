"""Reproducible measurements and readable figures for lab parts 1 and 2."""
import random
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .drag import DRAG


def measure_search(search, data, m=50, top_k=5, repeats=3):
    """Median runtime on identical data, with a reproducible random visit order."""
    times, reference = [], None
    for _ in range(repeats):
        random.seed(42)
        np.random.seed(42)
        start = perf_counter()
        result = np.asarray(search(data, m, top_k), dtype=float).reshape(-1, 2)
        times.append(perf_counter() - start)
        if reference is None:
            reference = result
        else:
            np.testing.assert_allclose(result, reference, atol=1e-9)
    return reference, float(np.median(times))


def result_table(result, offset=0):
    frame = pd.DataFrame(result, columns=['Начало окна', 'Расстояние до ближайшего соседа'])
    frame['Начало окна'] = frame['Начало окна'].astype(int)
    if offset:
        frame.insert(1, 'Индекс в исходном файле', frame['Начало окна'] + offset)
    return frame


def plot_discords(data, indices, m, title, ax=None, boundary=None):
    """Highlight a union of intervals once, avoiding dark repeated shading."""
    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(14, 3.4))
    ax.plot(data, color='#2878b5', linewidth=1, label='Временной ряд')
    indices = sorted(int(i) for i in indices)
    intervals = []
    for start in indices:
        stop = min(len(data), start + m)
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], stop)
        else:
            intervals.append([start, stop])
    for number, (start, stop) in enumerate(intervals):
        ax.axvspan(start, stop - 1, color='#e45756', alpha=0.22,
                   label='Выделенные окна' if number == 0 else None)
        ax.plot(np.arange(start, stop), data[start:stop], color='#c33f3f', linewidth=1.3)
    if boundary is not None:
        ax.axvline(boundary, color='#b17b00', linestyle='--', linewidth=1.3,
                   label='Примерная смена активности')
    ax.set(title=f'{title} | m={m}, окон: {len(indices)}',
           xlabel='Индекс отсчета в выбранном фрагменте', ylabel='Значение')
    ax.grid(alpha=0.2)
    ax.legend(loc='upper right', fontsize=8)
    if created:
        ax.figure.tight_layout()
        plt.show()
    return ax


def plot_times(times, title):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(list(times), list(times.values()), color=['#2878b5', '#e59032'])
    ax.bar_label(bars, fmt='%.4f', padding=3)
    ax.set(title=title, ylabel='Время, с (медиана трех запусков)')
    ax.set_ylim(0, max(times.values()) * 1.22)
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    plt.show()


def parameter_grid(data, lengths, thresholds, boundary=None):
    rows, results = [], {}
    for m in lengths:
        for r in thresholds:
            start = perf_counter()
            indices, distances, neighbors = DRAG(data, m, r)
            results[m, r] = (indices, distances, neighbors)
            row = {'m': m, 'r': r, 'Количество окон': len(indices),
                   'Время, с': perf_counter() - start,
                   'Максимальное расстояние': max(distances, default=np.nan)}
            if boundary is not None:
                row['Только смена активности'] = bool(indices) and all(i <= boundary < i + m for i in indices)
            rows.append(row)
    return pd.DataFrame(rows), results


def plot_parameter_examples(data, results, settings, title, boundary=None):
    fig, axes = plt.subplots(len(settings), 1, figsize=(14, 2.8 * len(settings)), squeeze=False)
    for ax, (m, r) in zip(axes[:, 0], settings):
        plot_discords(data, results[m, r][0], m, f'{title}, r={r}', ax, boundary)
    fig.tight_layout()
    plt.show()
