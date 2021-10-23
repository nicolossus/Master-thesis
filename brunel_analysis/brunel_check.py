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

# Fixed model parameters
order = 2500    # -> NE=10,000 ; NI=2500 ; N_tot=12,500 ; CE=1000 ; CI=250
epsilon = 0.1   # connection probability
D = 1.5         # synaptic delay (ms)
T = 1000        # simulation time (ms)
N_rec = 20      # record output from N_rec neurons
n_type = 'exc'  # record excitatory spike trains

# NEST settings
threads = 16        # number of threads to use in simulation
print_time = True   # print simulated time or not

# True parameter values
# g = 4 corresponds to balance between excitation and inhibition
J = 0.1         # excitatory synapse weight (mV)

g_ai = 4       # range [4, 8], use uniform prior
eta_ai = 1.5      # range [1, 3.5], use uniform prior

# simulator model
bnet = nm.models.BrunelNet(order=order,
                           epsilon=epsilon,
                           D=D,
                           J=J,
                           T=T,
                           N_rec=N_rec,
                           n_type=n_type,
                           print_time=print_time,
                           threads=threads
                           )

spiketrains = bnet(eta=eta_ai, g=g_ai)

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

t_start = 300. * pq.ms
t_stop = 800 * pq.ms

spiketrains_slice = slice_spiketrains(spiketrains,
                                      t_start=t_start,
                                      t_stop=t_stop
                                      )

# Raster plot
fig, ax = plt.subplots(figsize=(8, 4),
                       constrained_layout=True,
                       dpi=120
                       )

rasterplot_rates(spiketrains_slice, ax=ax)
ax.set(yticks=[1, 5, 10, 15, 20],
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
