#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import quantities as pq
import torch
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference.base import infer
from utils import *

'''
NEST does not support torch tensors which SBI requires. It is therefore
not possible at the current time to use SBI and NEST together.

nest.lib.hl_api_exceptions.PyNESTError: unknown Python type: <class 'torch.Tensor'>
'''

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


def simulator(params):
    """
    Returns summary statistics from conductance values in `params`.

    Summarizes the output of the HH simulator and converts it to `torch.Tensor`.
    """
    params = params.numpy()
    sim = bnet(*params)
    sum_stats = torch.as_tensor(sts(sim))
    return sum_stats


#params = [2., 5.]
#params = torch.tensor([2., 5.])
#sum_stats = simulator(params)
# print(sum_stats)

# priors
# eta: U(1.5, 4)
# g: U(1.5, 8)

prior_min = [1.5, 1.5]
prior_max = [4., 8.]

prior = utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),
                                    high=torch.as_tensor(prior_max)
                                    )

# inference
posterior = infer(simulator,
                  prior,
                  method='SNPE',
                  num_simulations=1000,
                  num_workers=8
                  )

save_posterior(posterior, 'data/brunel_posterior.pkl')
