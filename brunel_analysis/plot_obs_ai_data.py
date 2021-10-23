#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import quantities as pq
import seaborn as sns
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import correlation_coefficient
from utils import *
from viziphant.rasterplot import rasterplot_rates
from viziphant.spike_train_correlation import plot_corrcoef

# Set plot style
sns.set(context="paper", style='darkgrid', rc={"axes.facecolor": "0.96"})

# load observed data
spiketrains = load_spiketrain('data/obs_ai_data.pkl')

T = 1000
N_rec = 20

# statistics calculator
s_stats = ["mean_firing_rate",  # rate estimation
           "mean_cv",           # spike interval statistic
           "fanofactor"         # statistic across spike trains
           ]

t_start = 100. * pq.ms
t_stop = T * pq.ms

sts = nm.statistics.SpikeTrainStats(stats=s_stats,
                                    t_start=t_start,
                                    t_stop=t_stop
                                    )

sum_stats = sts(spiketrains)

for s, r in zip(s_stats, sum_stats):
    print(f'{s}: {r}')


spiketrains_slice = slice_spiketrains(spiketrains,
                                      t_start=t_start,
                                      t_stop=500 * pq.ms
                                      )

# Raster plot
fig, ax = plt.subplots(figsize=(8, 4),
                       constrained_layout=True,
                       dpi=120
                       )

rasterplot_rates(spiketrains_slice, ax=ax)
ax.set(yticks=range(1, N_rec + 1, 4),
       ylabel='Neuron'
       )

# Correlation plot
fig, ax = plt.subplots(figsize=(6, 4),
                       tight_layout=True,
                       dpi=120
                       )

binned_spiketrains = BinnedSpikeTrain(spiketrains_slice, bin_size=10 * pq.ms)
corrcoef_matrix = correlation_coefficient(binned_spiketrains)

plot_corrcoef(corrcoef_matrix, axes=ax)

ax.set(xlabel='Neuron',
       ylabel='Neuron',
       title='Correlation coefficient matrix'
       )

plt.show()
