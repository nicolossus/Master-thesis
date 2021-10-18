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
           "average_AHP_depth"]

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

# Rejection ABC sampler
sampler = pylfi.RejABC(obs_data,
                       hh,
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
filename = f'hh_rej_normal_best_posterior_org.jnl'
journal.save(path + filename)

# Local linear regression adjustment
journal = sampler.reg_adjust(method="loclinear",
                             transform=True,
                             return_journal=True
                             )

# Save journal
filename = f'hh_rej_normal_best_posterior_reg.jnl'
journal.save(path + filename)
