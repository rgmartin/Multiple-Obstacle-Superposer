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
from auxiliary_functions import compute_steepness, find_peaks, plot_surface_peaks
pio.renderers.default = 'notebook'

matlab_folder = r'C:\Users\Rubert\OneDrive - McGill University\PhD\Work done\7- Supervision\SURE - 2021\2 - Scripts\2021_07_makarenko_daphne_outputs\perturbation_results_vec'
os.chdir(matlab_folder)

matlab_files = ['R0_10A0_10B0_10Lambda1_0XM20_0Dx0_2Zc1N8.mat',
                'R0_20A0_20B0_20Lambda1_0XM20_0Dx0_1Zc1N8.mat',
                'R0_30A0_30B0_30Lambda1_0XM20_0Dx0_2Zc1N8.mat',
                'R0_40A0_40B0_40Lambda1_0XM10_0Dx0_1Zc1N8.mat',
                'R0_50A0_50B0_50Lambda1_0XM40_0Dx0_2Zc1N8.mat',
                'R0_60A0_60B0_60Lambda1_0XM20_0Dx0_2Zc1N8.mat',
                'R0_70A0_70B0_70Lambda1_0XM20_0Dx0_2Zc1N8.mat',
                'R0_80A0_80B0_80Lambda1_0XM40_0Dx0_2Zc1N8.mat',
                'R0_90A0_90B0_90Lambda1_0XM20_0Dx0_2Zc1N8.mat'                
               ]

r_range = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
separation_factor_range = np.array([3,4,5,10])
t_aster_range = np.linspace(0,1.2,60)
x_range = np.linspace(-10,10,1000)
index = pd.MultiIndex.from_product([r_range,separation_factor_range, t_aster_range,x_range], names =('r*','m_factor','t*','x'))

analysis = MultipleObstacleAnalysis()
analysis.set_no_of_obstacles(21)
df = pd.DataFrame({'eta': pd.Series(dtype='float')},index=index )
for r_aster, file in zip(r_range, matlab_files):
    analysis.read_matlab_file(file)
    analysis.set_x_vals(x_range)
    for separation_factor in separation_factor_range:
        analysis.set_separation(separation_factor * r_aster)
        for t_aster in t_aster_range:
            df.loc[r_aster,separation_factor,t_aster]['eta'] = analysis.eta(t_aster=t_aster,
                                                                            highest_order=8, smooth_window_size = 11)
df = df.reset_index()


fig1 = px.line(df[df['r*'].isin(r_range[:5])], x = 'x', y = 'eta', animation_frame = 't*', range_y=[-1,1], facet_col = 'r*',
              facet_row ='m_factor', title='Multiple obstacle surface perturbation',facet_row_spacing = 0.05,
               width = 1000, height = 600,)

fig1.show()

fig2 = px.line(df[df['r*'].isin(r_range[5:])], x = 'x', y = 'eta', animation_frame = 't*', range_y=[-1,1], facet_col = 'r*',
              facet_row ='m_factor', title='Multiple obstacle surface perturbation',facet_row_spacing = 0.05,
              width = 1000, height = 600)

fig2.show()


# TODO: plot a symetrical plot between -m and m
# TODO: create functions for all the sections of this notebook
# parameters to test:
m_factor = 3
r_aster = 0.3
x=x_range
plot_surface_peaks(df, m_factor, r_aster, x)


steepness_df = df[['m_factor','r*','t*']].drop_duplicates().reset_index()

s_central_vec = []
s_depression_vec = []
for m_factor in df['m_factor'].unique():
    for r in df['r*'].unique():
        for t in df['t*'].unique():
            filtr = (df['m_factor'] == m_factor) & (df['r*'] == r)  &  (df['t*'] == t)
            sub_df = df[filtr]
            x = sub_df['x']
            eta = sub_df['eta']
            s_central, s_depression = compute_steepness(x.values, eta.values, m_factor*r)
            s_central_vec.append(s_central)
            s_depression_vec.append(s_depression)
steepness_df['s_depression'] = s_depression_vec
steepness_df['s_central'] = s_central_vec

fig1 = px.line(steepness_df, x = 't*', y = ['s_central','s_depression'],  facet_col = 'r*', facet_row = 'm_factor',
                  width = 1000, height = 600, facet_row_spacing = 0.05, range_y = [0,0.5])
fig1.show()