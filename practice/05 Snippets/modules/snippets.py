import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from modules.analysis import snippet_labels, ACTIVITY_NAMES


def _finish(fig):
    fig.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('#E0E0E0' if ax.get_ylabel() == 'Кластер' else 'white')
        ax.tick_params(colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        ax.grid(alpha=.16, color='#777777')
    plt.show()


def plot_snippets(ts, snippets, timestamps=None, title='Найденные сниппеты'):
    """Plot representatives and regime labels aligned with window centers.

    Edge points without a complete centered window are gray, not assigned to cluster 0.
    """
    ts = np.asarray(ts, dtype=float)
    time = np.arange(len(ts)) if timestamps is None else np.asarray(timestamps)
    if time.shape != ts.shape:
        raise ValueError('Timestamps must match the time series.')
    labels = snippet_labels(snippets)
    m, k = snippets[0].shape[1], len(snippets[0])
    if len(labels) != len(ts) - m + 1:
        raise ValueError('Profile length must equal n-m+1.')
    centers = np.arange(len(labels)) + m // 2
    colors = plt.get_cmap('tab10').colors
    cmap = ListedColormap([colors[i % 10] for i in range(k)])
    fig, (main, strip) = plt.subplots(2, 1, figsize=(13, 4.8), sharex=True,
        gridspec_kw={'height_ratios': [7, 1]}, layout='constrained')
    main.plot(time, ts, color='#999999', linewidth=.7)
    for idx, start in enumerate(snippets[1]):
        stop = int(start) + m
        main.plot(time[start:stop], ts[start:stop], color=colors[idx % 10], linewidth=1.5,
                  label=f'Сниппет {idx}: {snippets[3][idx]:.1%} окон')
    main.set(title=title, ylabel='Сигнал' if timestamps is None else 'Угловая скорость, рад/с')
    main.legend(fontsize=9, loc='upper right')
    dt = np.median(np.diff(time))
    strip.set_facecolor('#E0E0E0')
    strip.imshow(labels[np.newaxis], cmap=cmap, norm=BoundaryNorm(np.arange(k+1)-.5, k),
                 aspect='auto', interpolation='nearest',
                 extent=[time[centers[0]]-dt/2, time[centers[-1]]+dt/2, 0, 1])
    strip.set(yticks=[], ylabel='Кластер', xlabel='Индекс точки' if timestamps is None else 'Время, с')
    strip.set_xlim(time[0], time[-1])
    _finish(fig)
    strip.set_facecolor('#E0E0E0')
    return main


def plot_activity_comparison(time, truth, predictions, title):
    fig, axes = plt.subplots(len(predictions)+1, 1, figsize=(13, 2.0+1.25*len(predictions)),
                             sharex=True, layout='constrained')
    classes = sorted(ACTIVITY_NAMES)
    rows = [('Истинная разметка', truth)] + list(predictions.items())
    for ax, (name, values) in zip(axes, rows):
        y = np.searchsorted(classes, values)
        ax.step(time, y, where='mid', linewidth=1.0)
        ax.set(title=name, yticks=range(len(classes)),
               yticklabels=['Подъем', 'Спуск'], ylim=(-.15, len(classes)-.85))
    axes[0].set_title(title + '\nИстинная разметка')
    axes[-1].set_xlabel('Время центра окна, с')
    _finish(fig)


def plot_parameter_selection(areas, changes, selected_k):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained')
    for row, name in enumerate(['X', 'Y', 'Z']):
        axes[row, 0].bar(areas.index, areas[name], color='#2463A5')
        axes[row, 0].set(title=f'Ось {name}: ProfileArea', ylabel='Площадь', xticks=areas.index, xlabel='k')
        valid = changes[name].dropna()
        axes[row, 1].bar(valid.index, valid, color='#D66B12')
        axes[row, 1].set(title=f'Ось {name}: change', ylabel='Относительное улучшение',
                          xticks=areas.index, xlabel='k')
        axes[row, 1].annotate('k=1: нет Area₀', (1, 0), xytext=(1, .02), fontsize=8)
    _finish(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5), layout='constrained')
    axes[0].bar(areas.index, areas['Среднее XYZ'], color='#2463A5')
    axes[0].set(title='Средний ProfileArea трех осей', ylabel='Площадь', xticks=areas.index, xlabel='k')
    valid = changes['Среднее XYZ'].dropna()
    axes[1].bar(valid.index, valid, color=['#C52E32' if k==selected_k else '#D66B12' for k in valid.index])
    axes[1].set(title=f'change для трех осей: выбран k={selected_k}', ylabel='Относительное улучшение',
                xticks=areas.index, xlabel='k')
    axes[1].annotate('k=1 не определен', (1, 0), xytext=(1, .02), fontsize=8)
    _finish(fig)
