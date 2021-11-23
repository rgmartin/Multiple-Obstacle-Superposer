# library imports
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
from Shot import Makarenko_Analysis, MultipleObstacleAnalysis
import plotly.io as pio
from scipy.signal import find_peaks, savgol_filter

pio.renderers.default = 'notebook'


# function to plot the surface and relevant properties at given parameters
def test_surface_peaks(x, y, m):
    # first separate the depression peak (which is centered at m/2)
    depression_flag = (0 <= x) & (x <= m)
    x_depression, y_depression = x[depression_flag], y[depression_flag]
    # find the minimum and its symmetrical opposite
    min_idx = np.argmin(y_depression)
    min_idx_sym = len(y_depression) - min_idx - 1
    min_idx, min_idx_sym = np.sort([min_idx, min_idx_sym])

    # plot the depression part
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            visible=False,
            x=np.concatenate([-x_depression,[-x_depression[min_idx]], x_depression]),
            y=np.concatenate([y_depression[::-1],[np.NaN], y_depression]),
            mode='lines',
            name='Depression'
        )
    )




    # check if the minimum is reached too close to the midpoint
    # in which case is is either a very small jet or there is no jet at all
    if np.abs(min_idx - len(y_depression) / 2) < 5:
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=[],
                y=[],
                mode='lines',
                name='Prominence'
            )
        )
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=[],
                y=[],
                mode='lines',
                name='Width'
            )
        )
    else:
        prominence = np.max(y_depression[min_idx:min_idx_sym + 1]) - np.min(y_depression)
        width = x_depression[min_idx_sym] - x_depression[min_idx]
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=np.ones(10) * m / 2,
                y=np.linspace(min(y_depression), min(y_depression) + prominence, 10),
                mode='lines',
                name='Prominence Depression'
            )
        )
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=np.linspace(x_depression[min_idx], x_depression[min_idx] + width, 10),
                y=np.ones(10) * min(y_depression),
                mode='lines',
                name='Width Depression'
            )
        )

    if not (np.abs(min_idx - len(y_depression) / 2) < 5):
        prominence = np.max(y_depression[min_idx:min_idx_sym + 1]) - np.min(y_depression)
        width = x_depression[min_idx_sym] - x_depression[min_idx]
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=x_depression[[min_idx, min_idx_sym]],
                y=y_depression[[min_idx, min_idx_sym]],
                mode='markers',
                marker=dict(
                    size=3,
                    color='red',
                    symbol='cross'
                ),
                name='Depression minima'
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                visible=False,
                x=[],
                y=[],
                mode='markers',
                marker=dict(
                    size=3,
                    color='red',
                    symbol='cross'
                ),
                name='Depression minima'
            )
        )

        # now separate the central peak

    central_flag = ((-x_depression[min_idx]) <= x) & (x <= x_depression[min_idx])
    x_central, y_central = x[central_flag], y[central_flag]
    # plot the central part
    fig.add_trace(
        go.Scatter(
            visible=False,
            x=x_central,
            y=y_central,
            mode='lines',
            name='Central'
        )
    )
    # compute slenderness
    prominence = np.max(y_central) - np.min(y_central)
    width = 2 * x_depression[min_idx]
    fig.add_trace(
        go.Scatter(
            visible=False,
            x=np.zeros(10),
            y=np.linspace(min(y_depression), min(y_depression) + prominence, 10),
            mode='lines',
            name='Prominence Central'
        )
    )
    fig.add_trace(
        go.Scatter(
            visible=False,
            x=np.linspace(-x_depression[min_idx], -x_depression[min_idx] + width, 10),
            y=np.ones(10) * min(y_depression),
            mode='lines',
            name='Width Central'
        )
    )
    return fig


# Function to plot the multiple peaks accross time
def plot_surface_peaks(df, m_factor, d_aster, x):
    fig = go.Figure()
    t_aster_range = df['t*'].unique()
    for t_aster in t_aster_range:
        y = df[(df['m*'] == m_factor) & (df['d*'] == d_aster) & (df['t*'] == t_aster)]['eta'].values
        r_aster = 1/(1+d_aster)
        fig_step = test_surface_peaks(x, y, m_factor * r_aster)
        for d in fig_step.data:
            fig.add_trace(d)
    steps = []
    for i in range(0, len(fig.data), len(fig_step.data)):
        step = dict(
            method="restyle",
            args=["visible", [False] * len(fig.data)],
        )
        step["label"] = "{:.2f}".format(t_aster_range[int(i / len(fig_step.data))])
        for j in range(len(fig_step.data)):
            step["args"][1][i + j] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    for j in range(len(fig_step.data)):
        fig.data[j].visible = True  # Toggle i'th trace to "visible"
    sliders = [dict(active=10, steps=steps, )]
    fig.layout.update(
        yaxis_range=[-0.5, 0.5],
        xaxis_title=r'$x$',
        yaxis_title=r'$\eta$',
        sliders=sliders)
    return fig


def compute_steepness(x, y, m):
    # first separate the depression peak (which is centered at m/2)
    depression_flag = (0 <= x) & (x <= m)
    x_depression, y_depression = x[depression_flag], y[depression_flag]

    # find the minimum and its symmetrical opposite
    min_idx = np.argmin(y_depression)
    min_idx_sym = len(y_depression) - min_idx - 1
    min_idx, min_idx_sym = np.sort([min_idx, min_idx_sym])

    # compute slenderness
    # check if the minimum is reached too close to the midpoint
    # in which case is is either a very small jet or there is no jet at all
    if np.abs(min_idx - len(y_depression) / 2) < 5:
        s_depression = 0
    else:
        prominence = np.max(y_depression[min_idx:min_idx_sym + 1]) - np.min(y_depression)
        width = x_depression[min_idx_sym] - x_depression[min_idx]
        s_depression = prominence / width

        # now separate the central peak

    central_flag = (-x_depression[min_idx] <= x) & (x <= x_depression[min_idx])
    x_central, y_central = x[central_flag], y[central_flag]

    # compute slenderness
    prominence = np.max(y_central) - np.min(y_central)
    width = 2 * x_depression[min_idx]
    s_central = prominence / width

    return s_central, s_depression
