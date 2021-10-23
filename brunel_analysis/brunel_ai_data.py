#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import pandas as pd
import pylfi
import quantities as pq
from tqdm import tqdm
from utils import *

# Fixed model parameters
order = 2500    # -> NE=10,000 ; NI=2500 ; N_tot=12,500 ; CE=1000 ; CI=250
epsilon = 0.1   # connection probability
D = 1.5         # synaptic delay (ms)
T = 1000        # simulation time (ms)
N_rec = 20      # record output from N_rec neurons
n_type = 'exc'  # record excitatory spike trains
J = 0.1         # excitatory synapse weight (mV)

# NEST settings
threads = 16        # number of threads to use in simulation
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

# priors
# ground truth
# eta_ai = 2.       # range [1, 4], use uniform prior
# g_ai = 5          # range [4, 8], use uniform prior
# In phase diagram:
# g = 4 corresponds to balance between excitation and inhibition
# eta = 1 corresponds to input sufficient to reach firing threshold

eta_prior = pylfi.Prior('uniform',
                        loc=1.5,
                        scale=2.5,
                        name='eta',
                        tex=r"$\eta$"
                        )

g_prior = pylfi.Prior('uniform',
                      loc=4,
                      scale=4,
                      name='g',
                      tex=r"$g$"
                      )

# gather data
N_sims = 2000
sum_stats = []
gs = g_prior.rvs(N_sims)
etas = eta_prior.rvs(N_sims)

sum_stats = []
for eta, g in tqdm(zip(etas, gs), total=N_sims):
    spiketrains = bnet(eta=eta, g=g)
    sum_stats.append(sts(spiketrains))

# store results in DataFrame
data = dict(zip(s_stats, np.stack(sum_stats, axis=-1)))
df = pd.DataFrame.from_dict(data)
df.insert(0, "eta", etas)
df.insert(1, "g", gs)
df.insert(2, r"$\eta$", etas)
df.insert(3, r"$g$", gs)
#df.replace([np.inf, -np.inf], np.nan, inplace=True)
# df.dropna(inplace=True)
df.to_csv('data/brunel_ai_data.csv', index=False)
print(df)
