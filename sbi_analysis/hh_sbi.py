#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer
from utils import *

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

# priors
prior_min = [31., 115.]
prior_max = [42., 126.]

prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                    high=torch.as_tensor(prior_max)
                                    )

# inference
posterior = infer(simulator,
                  prior,
                  method='SNPE',
                  num_simulations=1000,
                  num_workers=16
                  )

save_posterior(posterior, 'data/hh_posterior.pkl')

# check inference
s_obs = simulator(true_params)

samples = posterior.sample((1000,), x=s_obs)

fig, axes = analysis.pairplot(samples,
                              figsize=(5, 5),
                              points=true_params,
                              labels=labels_params,
                              points_offdiag={'markersize': 6},
                              points_colors='r')

plt.show()
