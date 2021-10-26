#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import quantities as pq
import seaborn as sns
import torch
from sbi import analysis as analysis
from utils import *

sns.set(context="paper", style='darkgrid', rc={"axes.facecolor": "0.96"})

# Fixed model parameters
order = 2500    # -> NE=10,000 ; NI=2500 ; N_tot=12,500 ; CE=1000 ; CI=250
epsilon = 0.1   # connection probability
D = 1.5         # synaptic delay (ms)
T = 1000        # simulation time (ms)
N_rec = 20      # record output from N_rec neurons
n_type = 'exc'  # record excitatory spike trains
J = 0.1         # excitatory synapse weight (mV)

# NEST settings
threads = 8        # number of threads to use in simulation
print_time = False  # print simulated time or not

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

# statistics calculator
s_stats = ["mean_firing_rate",  # rate estimation
           "mean_cv",           # spike interval statistic
           "fanofactor"         # statistic across spike trains
           ]

t_start = 100. * pq.ms  # record after 100 ms to avoid transient effects
t_stop = T * pq.ms

sts = nm.statistics.SpikeTrainStats(stats=s_stats,
                                    t_start=t_start,
                                    t_stop=t_stop
                                    )


# true parameters and respective labels
true_params = np.array([2., 5.])
labels_params = [r'$\eta$', r'$g$']

obs = bnet(*true_params)
s_obs = sts(obs)

posterior = load_posterior('data/brunel_posterior.pkl')

# check inference
samples = posterior.sample((1000,), x=s_obs)

fig, axes = analysis.pairplot(samples,
                              figsize=(5, 5),
                              points=true_params,
                              labels=labels_params,
                              upper=['kde'],
                              diag=['kde'],
                              points_offdiag={'markersize': 6},
                              points_colors='r',
                              )

plt.show()

# post_samples = posterior.sample((50,), x=s_obs).numpy()
# print(post_samples)
