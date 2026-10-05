import numpy as np
import pandas as pd
import io
import json
import base64
import matplotlib.pyplot as plt
from IPython.display import display

# for visualization
def _show(fig):
    """Store Plotly and a PNG fallback together for VS Code and GitHub."""
    fig.update_layout(template='plotly_white', paper_bgcolor='white',
                      plot_bgcolor='white', font=dict(color='black'))
    if any(trace.type != 'scatter' for trace in fig.data):
        fig.show(renderer='plotly_mimetype')
        return
    xrefs = sorted(set(trace.xaxis or 'x' for trace in fig.data))
    with plt.rc_context({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'text.color': 'black', 'axes.labelcolor': 'black',
                         'xtick.color': 'black', 'ytick.color': 'black',
                         'axes.edgecolor': 'black', 'font.size': 10}):
        static, axes = plt.subplots(1, len(xrefs), figsize=(11, 4.5), squeeze=False,
                                   gridspec_kw={'width_ratios': [1, 4]} if len(xrefs) == 2 else None)
        for trace in fig.data:
            ax = axes[0, xrefs.index(trace.xaxis or 'x')]
            ax.plot(np.asarray(trace.x), np.asarray(trace.y),
                    color=trace.line.color, linewidth=trace.line.width or 2,
                    label=str(trace.name) if trace.name is not None else None)
        for ref, ax in zip(xrefs, axes[0]):
            suffix = ref[1:]
            xaxis = getattr(fig.layout, 'xaxis' + suffix)
            yaxis = getattr(fig.layout, 'yaxis' + suffix)
            ax.set_xlabel(xaxis.title.text or 'Time')
            ax.set_ylabel(yaxis.title.text or 'Values')
            ax.grid(alpha=0.2)
            if any(line.get_label() and not line.get_label().startswith('_') for line in ax.lines):
                ax.legend(fontsize=8, loc='best')
        if len(xrefs) == 2:
            for ax, annotation in zip(axes[0], fig.layout.annotations):
                ax.set_title(annotation.text)
        static.suptitle(fig.layout.title.text or '')
        static.tight_layout()
        buffer = io.BytesIO()
        static.savefig(buffer, format='png', dpi=150, facecolor='white')
        plt.close(static)
    display({'application/vnd.plotly.v1+json': json.loads(fig.to_json()),
             'image/png': base64.b64encode(buffer.getvalue()).decode('ascii')}, raw=True)


import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px



def plot_ts_set(ts_set: np.ndarray, title: str = 'Input Time Series Set') -> None:
    """
    Plot the time series set

    Parameters
    ----------
    ts_set: time series set
    title: title of plot
    """

    ts_num, m = ts_set.shape

    fig = go.Figure()

    for i in range(ts_num):
        fig.add_trace(go.Scatter(x=np.arange(m), y=ts_set[i], line=dict(width=3), name="Time series " + str(i)))

    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)
    fig.update_layout(title=title,
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black'))
                      )

    _show(fig)


def mplot2d(x: np.ndarray, y: np.ndarray, plot_title: str = None, x_title: str = None, y_title: str = None, trace_titles: np.ndarray = None) -> None:
    """
    Multiple 2D Plots on figure for different experiments

    Parameters
    ----------
    x: values of x axis of plot
    y: values of y axis of plot
    plot_title: title of plot
    x_title: title of x axis of plot
    y_title: title of y axis of plot
    trace_titles: titles of plot traces (lines)
    """

    fig = go.Figure()

    for i in range(y.shape[0]):
        fig.add_trace(go.Scatter(x=x, y=y[i], line=dict(width=3), name=trace_titles[i]))

    fig.update_xaxes(showgrid=False,
                     title=x_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2,
                     tickvals=x)
    fig.update_yaxes(showgrid=False,
                     title=y_title,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks='outside',
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black')),
                      width=1000,
                      height=600
                      )

    _show(fig)


def plot_bestmatch_data(ts: np.ndarray, query: np.ndarray) -> None:
    """
    Visualize the input data (time series and query) for the best match task

    Parameters
    ----------
    ts: time series
    query: query
    """

    query_len = query.shape[0]
    ts_len = ts.shape[0]

    fig = make_subplots(rows=1, cols=2, column_widths=[0.1, 0.9], subplot_titles=("Query", "Time Series"), horizontal_spacing=0.04)

    fig.add_trace(go.Scatter(x=np.arange(query_len), y=query, line=dict(color=px.colors.qualitative.Plotly[1])),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(ts_len), y=ts, line=dict(color=px.colors.qualitative.Plotly[0])),
                row=1, col=2)

    fig.update_annotations(font=dict(size=24, color='black'))

    fig.update_xaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)
    fig.update_yaxes(showgrid=False,
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1,
                     mirror=True)

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      showlegend=False,
                      title_x=0.5)

    _show(fig)


def plot_bestmatch_results(ts: np.ndarray, query: np.ndarray, bestmatch_results: dict) -> None:
    """
    Visualize the best match results

    Parameters
    ----------
    ts: time series
    query: query
    bestmatch_results: output data found by the best match algorithm
    """

    m = len(query)
    indices = bestmatch_results['indices']
    distances = bestmatch_results['distances']
    color = px.colors.qualitative.Plotly[1]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.2, 0.8],
                        subplot_titles=('Query', 'Best matches in the time series'))
    fig.add_trace(go.Scatter(x=np.arange(m), y=query, line=dict(color=color, width=3),
                             name='Query'), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(len(ts)), y=ts,
                             line=dict(color=px.colors.qualitative.Plotly[0]),
                             name='ECG'), row=1, col=2)
    for rank, (idx, distance) in enumerate(zip(indices, distances), start=1):
        fig.add_trace(go.Scatter(x=np.arange(idx, idx+m), y=ts[idx:idx+m],
                                 line=dict(color=color, width=3),
                                 name=f'Match {rank}: i={idx}, d={distance:.4f}'),
                      row=1, col=2)
    fig.update_xaxes(title_text='Time')
    fig.update_yaxes(title_text='Values')
    fig.update_layout(title='Subsequence search results', width=1100, height=450,
                      legend=dict(orientation='h', y=-0.2))
    _show(fig)


def pie_chart(labels: np.ndarray, values: np.ndarray, plot_title='Pie chart') -> None:
    """
    Build the pie chart

    Parameters
    ----------
    labels: sector labels
    values: values
    """

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

    fig.update_traces(textfont_size=20)
    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'xanchor': 'center'},
                      title_font=dict(size=24, color='black'),
                      legend=dict(font=dict(size=20, color='black')),
                      width=700,
                      height=500
                      )

    _show(fig)
