#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import seaborn as sns
import torch
from sbi import analysis as analysis
from utils import *

sns.set(context="paper", style='darkgrid', rc={"axes.facecolor": "0.96"})

# Simulator model
T = 120           # Simulation time [ms]
dt = 0.025        # Time step [ms]
I_amp = 10        # Input stimuls amplitude [microA/cm^2]
t_stim_on = 10    # Stimulus onset [ms]
t_stim_off = 110  # Stimulus offset [ms]
stimulus = nm.stimulus.Constant(I_amp, t_stim_on, t_stim_off)
hh = nm.models.HodgkinHuxley(stimulus, T, dt)

# Summary statistics calculator
s_stats = ["average_AP_overshoot",
           "spike_rate",
           "average_AP_width",
           "average_AHP_depth",
           ]

sps = nm.statistics.SpikeStats(t_stim_on=t_stim_on,
                               t_stim_off=t_stim_off,
                               stats=s_stats
                               )


def simulator(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    sim = hh(*params)
    sum_stats = torch.as_tensor(sps(*sim))
    return sum_stats


# true parameters and respective labels
true_params = np.array([36., 120.])
labels_params = [r'$\bar{g}_\mathrm{K}$', r'$\bar{g}_\mathrm{Na}$']


posterior = load_posterior('data/hh_posterior.pkl')

# check inference
s_obs = simulator(true_params)

samples = posterior.sample((1000,), x=s_obs)

fig, axes = analysis.pairplot(samples,
                              figsize=(5, 5),
                              points=true_params,
                              labels=labels_params,
                              upper=['kde'],
                              diag=['kde'],
                              points_offdiag={'markersize': 6},
                              points_colors='r',
                              cmap='Spectral',
                              fig_bg_colors={"upper": 'C2',
                                             "diag": 'viridis',
                                             "lower": 'C1'}
                              )

plt.show()

# post_samples = posterior.sample((50,), x=s_obs).numpy()
# print(post_samples)
