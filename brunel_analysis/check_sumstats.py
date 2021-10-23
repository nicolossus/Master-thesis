#!/usr/bin/env python
# -*- coding: utf-8 -*-

import neuromodels as nm
import numpy as np
import quantities as pq
from utils import *

# Fixed model parameters
order = 2500    # -> NE=10,000 ; NI=2500 ; N_tot=12,500 ; CE=1000 ; CI=250
epsilon = 0.1   # connection probability
D = 1.5         # synaptic delay (ms)
T = 1000        # simulation time (ms)
N_rec = 20      # record output from N_rec neurons
n_type = 'exc'  # record excitatory spike trains

J = 0.35         # excitatory synapse weight (mV)

# NEST settings
threads = 16        # number of threads to use in simulation
print_time = False   # print simulated time or not

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

t_start = 100. * pq.ms
t_stop = T * pq.ms

sts = nm.statistics.SpikeTrainStats(stats=s_stats,
                                    t_start=t_start,
                                    t_stop=t_stop
                                    )

# Parameter values
# g = 4 corresponds to balance between excitation and inhibition
g_ai = [4, 5, 6]            # range [4, 8], use uniform prior
eta_ai = [1.5, 2, 2]      # range [1, 3.5], use uniform prior
# g_ai = [5, 6]            # range [4, 8], use uniform prior
# eta_ai = [2, 2]      # range [1, 3.5], use uniform prior


# load observed data
obs_spiketrains = load_spiketrain('data/obs_ai_data.pkl')

obs_sumstats = sts(obs_spiketrains)


print(f'{obs_sumstats=}')
print()

sim_sumstats_all = []

for eta, g in zip(eta_ai, g_ai):
    spiketrains = bnet(eta=eta, g=g)
    sim_sumstats = sts(spiketrains)
    sim_sumstats_all.append(sim_sumstats)
    print(f'{eta=}, {g=}')
    print(f'{sim_sumstats}')
    print()


def sd(a, axis=0):
    """Standard deviation from the mean"""
    a = np.asarray(a, dtype=float)
    #a = a.astype(np.float64)
    a[a == np.inf] = np.NaN
    sd = np.sqrt(np.nanmean(
        np.abs(a - np.nanmean(a, axis=axis))**2, axis=axis))
    return sd


def l2norm(s_sim, s_obs, scale=1):
    s_sim = np.asarray(sim_sumstats, dtype=float)
    s_obs = np.asarray(obs_sumstats, dtype=float)
    q = (s_sim - s_obs) / scale
    dist = np.linalg.norm(q, ord=2)
    return dist


scale = sd(sim_sumstats_all)
print(f"{scale=}")
print()

distances_unscaled = []
distances_scaled = []

for s_stat in sim_sumstats_all:
    dist_unscaled = l2norm(s_stat, obs_sumstats)
    dist_scaled = l2norm(s_stat, obs_sumstats, scale=scale)
    distances_unscaled.append(dist_unscaled)
    distances_scaled.append(dist_scaled)
    print(f"{dist_unscaled=}")
    print(f"{dist_scaled=}")
    print()

# accept 50%
quantile = 0.5
print(f"{distances_unscaled=}")
print(f"{distances_scaled=}")
distances_unscaled = np.array(distances_unscaled, dtype=np.float64)
distances_scaled = np.array(distances_scaled, dtype=np.float64)

threshold_unscaled = np.quantile(distances_unscaled, quantile)
threshold_scaled = np.quantile(distances_scaled, quantile)
print(f'{threshold_unscaled=}')
print(f'{threshold_scaled=}')
