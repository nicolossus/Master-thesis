#!/usr/bin/env python
# -*- coding: utf-8 -*-

import neuromodels as nm
from utils import *

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

g_ai = 5        # range [4, 8], use uniform prior
eta_ai = 2      # range [1, 3.5], use uniform prior

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

save_spiketrain(spiketrains, 'data/obs_ai_data.pkl')
