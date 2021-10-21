import neuromodels as nm
import numpy as np
import pandas as pd
import pylfi
from tqdm import tqdm

# Simulator model
T = 120           # Simulation time [ms]
dt = 0.025        # Time step [ms]
I_amp = 10        # Input stimuls amplitude [microA/cm^2]
t_stim_on = 10    # Stimulus onset [ms]
t_stim_off = 110  # Stimulus offset [ms]
stimulus = nm.stimulus.Constant(I_amp, t_stim_on, t_stim_off)
hh = nm.models.HodgkinHuxley(stimulus, T, dt)

# Summary statistics calculator
# spike statistics
s_stats = ["average_AP_overshoot",
           "spike_rate",
           "average_AP_width",
           "average_AHP_depth",
           "latency_to_first_spike",
           "accommodation_index"]

sps = nm.statistics.SpikeStats(t_stim_on=t_stim_on,
                               t_stim_off=t_stim_off,
                               stats=s_stats
                               )

# Study config
size = 2000  # samples from prior predictive
gbarK0 = 36.
gbarNa0 = 120.

# Normal priors
gbarKs = pylfi.Prior('norm',
                     loc=gbarK0,
                     scale=2,
                     name='gbarK'
                     ).rvs(size)

gbarNas = pylfi.Prior('norm',
                      loc=gbarNa0,
                      scale=2,
                      name='gbarNa'
                      ).rvs(size)


sum_stats = []
for gbarK, gbarNa in tqdm(zip(gbarKs, gbarNas), total=size):
    V, t = hh(gbar_K=gbarK, gbar_Na=gbarNa)
    sum_stats.append(sps(V, t))

data = dict(zip(s_stats, np.stack(sum_stats, axis=-1)))
df = pd.DataFrame.from_dict(data)
df.insert(0, "gbarK", gbarKs)
df.insert(1, "gbarNa", gbarNas)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.to_csv('data/sum_stats_prior_pred_normal.csv', index=False)

# Uniform priors
gbarKs = pylfi.Prior('uniform',
                     loc=31,
                     scale=10,
                     name='gbarK',
                     ).rvs(size)

gbarNas = pylfi.Prior('uniform',
                      loc=115,
                      scale=10,
                      name='gbarNa',
                      ).rvs(size)

sum_stats = []
for gbarK, gbarNa in tqdm(zip(gbarKs, gbarNas), total=size):
    V, t = hh(gbar_K=gbarK, gbar_Na=gbarNa)
    sum_stats.append(sps(V, t))

data = dict(zip(s_stats, np.stack(sum_stats, axis=-1)))
df = pd.DataFrame.from_dict(data)
df.insert(0, "gbarK", gbarKs)
df.insert(1, "gbarNa", gbarNas)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.to_csv('data/sum_stats_priorpred_uniform.csv', index=False)
