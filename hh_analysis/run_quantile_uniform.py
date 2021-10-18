import time

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
gbarK = pylfi.Prior('uniform',
                    loc=31,
                    scale=10,
                    name='gbarK',
                    tex=r"$\bar{g}_\mathrm{K}$"
                    )

gbarNa = pylfi.Prior('uniform',
                     loc=115,
                     scale=10,
                     name='gbarNa',
                     tex=r"$\bar{g}_\mathrm{Na}$"
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
quantile_lst = [0.1, 0.3, 0.5, 0.7, 0.9]
n_sim_pilot = 2000
#pilot_seed = 3461876
n_samples_posterior = 1000
trials = 10
# seed list must have same length as number of trials
#seed_lst = [7, 42, 99, 12531, 8963, 137654, 384936146, 13438, 3431, 7773432]

data = []

for quantile in quantile_lst:
    # run pilot study to set epsilon based on provided quantile
    sampler.pilot_study(n_sim_pilot,
                        quantile=quantile,
                        stat_scale='sd',
                        n_jobs=16,
                        # seed=pilot_seed
                        )

    for trial in range(trials):

        # start timer
        start_time = time.time()
        # sample from posterior
        journal = sampler.sample(n_samples_posterior,
                                 use_pilot=True,
                                 n_jobs=16,
                                 # seed=seed_lst[trial],
                                 return_journal=True
                                 )
        # end timer
        end_time = time.time() - start_time
        # store timer
        data.append([end_time, quantile])
        # save journal with original samples
        quantile_str = str(quantile).replace('.', '_')
        filename = f'hh_rej_uniform_org_quantile_{quantile_str}_run_{trial}.jnl'
        path = 'data/'
        journal.save(path + filename)

        # regression adjustment
        journal = sampler.reg_adjust(method="loclinear",
                                     transform=True,
                                     return_journal=True
                                     )

        # save journal with adjusted samples
        filename = f'hh_rej_uniform_reg_quantile_{quantile_str}_run_{trial}.jnl'
        journal.save(path + filename)


df = pd.DataFrame(data, columns=['time', 'quantile'])
df.to_csv('data/run_time_quantile.csv', index=False)
