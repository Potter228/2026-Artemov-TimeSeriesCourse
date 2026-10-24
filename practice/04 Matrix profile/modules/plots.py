import math
import numpy as np
import matplotlib.pyplot as plt


def _finish(fig):
    """Keep saved notebook figures readable independently of editor theme."""
    fig.set_facecolor('white')
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.title.set_color('black')
        for spine in ax.spines.values():
            spine.set_color('#555555')
        ax.grid(alpha=0.18, color='#777777')
    plt.show()


def plot_ts(ts: np.ndarray, title: str = 'Временной ряд') -> None:
    fig, ax = plt.subplots(figsize=(12, 3.5), layout='constrained')
    ax.plot(np.arange(len(ts)), ts, color='#2463A5', linewidth=0.8)
    ax.set(title=title, xlabel='Индекс отсчета или начала окна', ylabel='Значение')
    _finish(fig)


def plot_motifs(mp: dict, top_k_motifs: dict) -> None:
    pairs = top_k_motifs['indices']
    count = len(pairs)
    rows = 2 + max(1, math.ceil(count / 2))
    fig = plt.figure(figsize=(14, 3 * rows), layout='constrained')
    grid = fig.add_gridspec(rows, 2)
    series_ax = fig.add_subplot(grid[0, :])
    profile_ax = fig.add_subplot(grid[1, :])
    data, m = mp['data']['ts1'], mp['m']
    series_ax.plot(data, color='#AAAAAA', linewidth=0.7)
    profile_ax.plot(np.arange(len(mp['mp'])), mp['mp'], color='#777777', linewidth=0.7)
    colors = plt.get_cmap('tab10').colors
    for rank, ((left, right), distance) in enumerate(zip(pairs, top_k_motifs['distances'])):
        color = colors[rank % len(colors)]
        pair_ax = fig.add_subplot(grid[2 + rank // 2, rank % 2])
        for occurrence, idx in enumerate((left, right)):
            series_ax.plot(np.arange(idx, idx + m), data[idx:idx + m], color=color, linewidth=1.5)
            profile_ax.scatter(idx, mp['mp'][idx], color=color, marker='*', s=75, zorder=3)
            values = data[idx:idx + m]
            deviation = values.std()
            normalized = (values - values.mean()) / deviation if deviation > 0 else np.zeros(m)
            pair_ax.plot(normalized, color=color, linestyle='-' if occurrence == 0 else '--',
                         linewidth=1.1, label=f'Начало {idx}')
        pair_ax.set(title=f'Мотив {rank + 1}, расстояние {distance:.3f}',
                    xlabel='Индекс внутри окна', ylabel='z-нормализованное значение')
        pair_ax.legend(fontsize=9)
    series_ax.set(title=f'Исходный ряд и {count} пар мотивов', xlabel='Индекс отсчета', ylabel='Потребление')
    profile_ax.set(title='Матричный профиль и начала выбранных окон',
                   xlabel='Индекс начала окна', ylabel='Расстояние до соседа')
    _finish(fig)


def plot_discords(mp: dict, top_k_discords: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout='constrained', sharex=True)
    data, m = mp['data']['ts1'], mp['m']
    axes[0].plot(data, color='#2463A5', linewidth=0.8, label='Исходный ряд')
    for rank, idx in enumerate(top_k_discords['indices']):
        axes[0].plot(np.arange(idx, idx + m), data[idx:idx + m], color='#C52E32',
                     linewidth=1.2, label='Диссонансы' if rank == 0 else None)
    axes[0].set(title=f"Такси NY и top-{len(top_k_discords['indices'])} диссонансов", ylabel='Значение ряда')
    axes[0].legend()
    axes[1].plot(np.arange(len(mp['mp'])), mp['mp'], color='#2463A5', linewidth=0.8)
    axes[1].scatter(top_k_discords['indices'], top_k_discords['distances'],
                    color='#C52E32', marker='*', s=65, zorder=3)
    axes[1].set(title='Матричный профиль и выбранные максимумы',
                xlabel='Индекс начала окна', ylabel='Расстояние до соседа')
    _finish(fig)


def plot_segmentation(mp: dict, threshold: float) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), layout='constrained', sharex=True)
    data, values, m = mp['data']['ts1'], mp['mp'], mp['m']
    centers = np.arange(len(values)) + m // 2
    stairs = values < threshold
    axes[0].plot(data, color='#BBBBBB', linewidth=0.5)
    for mask, color, label in [(~stairs, '#2463A5', 'Ходьба'), (stairs, '#D66B12', 'Лестница')]:
        axes[0].plot(centers, np.where(mask, data[centers], np.nan),
                     color=color, linewidth=0.8, label=label)
    axes[0].set(title='Сегментация PAMAP по центрам окон', ylabel='Сигнал акселерометра')
    axes[0].legend()
    axes[1].plot(centers, values, color='#2463A5', linewidth=0.8, label='Матричный профиль')
    axes[1].axhline(threshold, color='#C52E32', linestyle='--', label=f'Порог {threshold:.3f}')
    axes[1].set(title='Ниже порога: повторяющаяся активность на лестнице',
                xlabel='Индекс центра окна', ylabel='Расстояние до соседа')
    axes[1].legend()
    _finish(fig)
