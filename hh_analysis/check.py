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

# Priors
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

# Study config
quantile = 0.5
n_sim_pilot = 160
pilot_seed = 3461876
n_samples_posterior = 1000
trials = 10

sampler.pilot_study(n_sim_pilot,
                    quantile=quantile,
                    stat_scale='sd',  # 'mad',
                    n_jobs=16,
                    seed=pilot_seed
                    )
print(sampler.epsilon)
