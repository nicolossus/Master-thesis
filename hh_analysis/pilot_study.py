import neuromodels as nm
import numpy as np
import pandas as pd
import pylfi

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
           "latency_to_first_spike",
           "accommodation_index"]

sps = nm.statistics.SpikeStats(t_stim_on=t_stim_on,
                               t_stim_off=t_stim_off,
                               stats=s_stats
                               )

# Observed data
gbarK0 = 36.
gbarNa0 = 120.

V, t = hh(gbar_K=gbarK0, gbar_Na=gbarNa0)
obs_data = (V, t)


# Study config
quantile = 0.5
n_sims_lst = [250, 500, 750, 1000, 1250, 1500, 1750, 2000]
seed_lst = [4242, 3833, 93249, 127531, 896793, 1376544, 221421, 74572]

headers = s_stats + ["epsilon", "n_sims"]

# Normal priors
gbarK = pylfi.Prior('norm',
                    loc=gbarK0,
                    scale=2,
                    name='gbarK'
                    )

gbarNa = pylfi.Prior('norm',
                     loc=gbarNa0,
                     scale=2,
                     name='gbarNa'
                     )

priors = [gbarK, gbarNa]

# Rejection ABC sampler
sampler = pylfi.RejABC(obs_data,
                       hh,
                       sps,
                       priors,
                       log=True
                       )

# Run pilot study
results = []

for n_sims, seed in zip(n_sims_lst, seed_lst):
    sampler.pilot_study(n_sims,
                        quantile=quantile,
                        stat_scale='sd',
                        n_jobs=16,
                        seed=seed
                        )
    stat_scales = sampler.stat_scale
    epsilon = sampler.epsilon
    results.append([*stat_scales, epsilon, n_sims])


data = dict(zip(headers, np.stack(results, axis=-1)))
df = pd.DataFrame.from_dict(data)
df.to_csv('data/pilot_normal.csv', index=False)

# Uniform priors
gbarK = pylfi.Prior('uniform',
                    loc=31,
                    scale=10,
                    name='gbarK'
                    )

gbarNa = pylfi.Prior('uniform',
                     loc=115,
                     scale=10,
                     name='gbarNa'
                     )

priors = [gbarK, gbarNa]

# Rejection ABC sampler
sampler = pylfi.RejABC(obs_data,
                       hh,
                       sps,
                       priors,
                       log=True
                       )

# Run pilot study
results = []

for n_sims, seed in zip(n_sims_lst, seed_lst):
    sampler.pilot_study(n_sims,
                        quantile=quantile,
                        stat_scale='sd',
                        n_jobs=16,
                        seed=seed
                        )
    stat_scales = sampler.stat_scale
    epsilon = sampler.epsilon
    results.append([*stat_scales, epsilon, n_sims])


data = dict(zip(headers, np.stack(results, axis=-1)))
df = pd.DataFrame.from_dict(data)

df.to_csv('data/pilot_uniform.csv', index=False)
