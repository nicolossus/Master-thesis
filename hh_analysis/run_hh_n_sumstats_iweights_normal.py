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

# Study config
path = 'data/'
n_samples_posterior = 1000
n_sims_pilot = 2000
quantile = 0.4
pilot_seeds = [8324, 23478234, 324, 984578, 45842874, 2797434]
sampler_seeds = [462434, 14, 796, 994522, 777428, 2452]

s_stats_all = ["average_AP_overshoot",
               "spike_rate",
               "average_AP_width",
               "average_AHP_depth",
               "latency_to_first_spike",
               "accommodation_index"]

df_weights = pd.read_csv('data/sumstat_weights_normal.csv', index_col=0)

for i in range(1, len(s_stats_all) + 1):

    # Summary statistics calculator
    s_stats = s_stats_all[:i]
    sps = nm.statistics.SpikeStats(t_stim_on=t_stim_on,
                                   t_stim_off=t_stim_off,
                                   stats=s_stats
                                   )

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
                        seed=pilot_seeds[i - 1]
                        )

    # Sample from posterior
    sampler.sample(n_samples_posterior,
                   use_pilot=True,
                   stat_weight=stat_weights,
                   n_jobs=16,
                   seed=sampler_seeds[i - 1],
                   return_journal=False
                   )

    # Local linear regression adjustment
    journal = sampler.reg_adjust(method="loclinear",
                                 transform=True,
                                 return_journal=True
                                 )

    # Save journal
    filename = f'hh_rej_normal_nsumstats_{i}_iweights.jnl'
    journal.save(path + filename)
