import matplotlib.pyplot as plt
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

noise_scale = 1.
stimulus_noisy = nm.stimulus.NoisyConstant(I_amp,
                                           t_stim_on,
                                           t_stim_off,
                                           noise_scale=noise_scale)
hh_noisy = nm.models.HodgkinHuxley(stimulus_noisy, T, dt)


# Observed data
gbarK0 = 36.
gbarNa0 = 120.

df = pd.read_csv('data/hh_noisy.csv')
V = df["V"].to_numpy()
t = df["t"].to_numpy()
obs_data = (V, t)

# Summary statistics calculator
s_stats = ["average_AP_overshoot",
           "spike_rate",
           "average_AP_width",
           "average_AHP_depth",
           # "latency_to_first_spike",
           # "accommodation_index"
           ]

sps = nm.statistics.SpikeStats(t_stim_on=t_stim_on,
                               t_stim_off=t_stim_off,
                               stats=s_stats
                               )

# Priors
gbarK = pylfi.Prior('norm',
                    loc=gbarK0,
                    scale=2,
                    name='gbarK',
                    tex=r"$\bar{g}_\mathrm{K}$"
                    )

gbarNa = pylfi.Prior('norm',
                     loc=gbarNa0,
                     scale=2,
                     name='gbarNa',
                     tex=r"$\bar{g}_\mathrm{Na}$"
                     )

priors = [gbarK, gbarNa]

# Inference config
path = "data/"
quantile = 0.4
n_sims_pilot = 2000
n_samples_posterior = 1000
seed_pilot = 7
seed_sampler = 42

df_weights = pd.read_csv('data/sumstat_weights_normal.csv', index_col=0)
# summary statistic weights
stat_weights = df_weights.loc[s_stats]["Weight"].to_numpy()
# Sum weights to 1
stat_weights /= np.sum(stat_weights)
print(stat_weights)

# Rejection ABC sampler
sampler = pylfi.RejABC(obs_data,
                       hh_noisy,
                       sps,
                       priors,
                       log=True
                       )

# Pilot study
sampler.pilot_study(n_sims_pilot,
                    quantile=quantile,
                    stat_scale='sd',
                    stat_weight=stat_weights,
                    n_jobs=16,
                    seed=seed_pilot
                    )

# Sample from posterior
journal = sampler.sample(n_samples_posterior,
                         use_pilot=True,
                         stat_weight=stat_weights,
                         n_jobs=16,
                         seed=seed_sampler,
                         return_journal=True
                         )
# Save journal
filename = f'hh_noise_rej_noisy_posterior_org.jnl'
journal.save(path + filename)

# Local linear regression adjustment
journal = sampler.reg_adjust(method="loclinear",
                             kernel='epkov',
                             transform=True,
                             return_journal=True
                             )

# Save journal
filename = f'hh_noise_rej_noisy_posterior_reg.jnl'
journal.save(path + filename)
