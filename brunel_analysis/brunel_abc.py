#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import pylfi
import quantities as pq
import seaborn as sns
from utils import *

# load observed data
spiketrains = load_spiketrain('data/obs_ai_data.pkl')

# simulator model
order = 2500  # 2500    # -> NE=10,000 ; NI=2500 ; N_tot=12,500 ; CE=1000 ; CI=250
epsilon = 0.1   # connection probability
D = 1.5         # synaptic delay (ms)
T = 1000        # simulation time (ms)
N_rec = 20      # record output from N_rec neurons
n_type = 'exc'  # record excitatory spike trains

# NEST settings
threads = 16          # number of threads to use in simulation
print_time = False   # print simulated time or not

# True parameter values
# g = 4 corresponds to balance between excitation and inhibition
J = 0.35         # excitatory synapse weight (mV)

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

# Priors
g = pylfi.Prior('uniform',
                loc=4,
                scale=2,
                name='g',
                tex=r"$g$"
                )

eta = pylfi.Prior('uniform',
                  loc=1,
                  scale=1,
                  name='eta',
                  tex=r"$\eta$"
                  )
'''
g = pylfi.Prior('uniform',
                loc=4,
                scale=4,
                name='g',
                tex=r"$g$"
                )

eta = pylfi.Prior('uniform',
                  loc=1,
                  scale=2.5,
                  name='eta',
                  tex=r"$\eta$"
                  )
'''
priors = [g, eta]

'''
class RABC:

    def __init__(self, obs, simulator, stat_calc, priors):
        self.obs = obs
        self.simulator = simulator
        self.stat_calc = stat_calc
        self.priors = priors

    def simulate(self):
        print("Calculate obs sum stats")
        obs_sumstats = self.stat_calc(self.obs)
        print("Done")

        print("Start loop")
        for i in range(5):
            print(f"{i=}")

            thetas = [prior.rvs() for prior in self.priors]
            print(f"{thetas=}")

            print("Begin simulation")
            sim = self.simulator(*thetas)
            print("Simulation done")

            print("Calculate sim sum stats")
            sim_sumstats = self.stat_calc(sim)
            print("Calculation done")


# Sanity check
thetas = [prior.rvs() for prior in priors]
sim = bnet(*thetas)
sum_stats = sts(sim)
print("Sanity check sum stats:", sum_stats)

rabc = RABC(spiketrains, bnet, sts, priors)
rabc.simulate()
'''
# Inference config

path = "data/"
quantile = 0.4
n_sims_pilot = 10
n_samples_posterior = 5
seed_pilot = 7
seed_sampler = 42


'''
df_weights = pd.read_csv('data/sumstat_weights_uniform.csv', index_col=0)
# summary statistic weights
stat_weights = df_weights.loc[s_stats]["Weight"].to_numpy()
# Sum weights to 1
stat_weights /= np.sum(stat_weights)
'''

# Rejection ABC sampler

sampler = pylfi.RejABC(spiketrains,
                       bnet,
                       sts,
                       priors,
                       log=True
                       )
'''
# Pilot study
sampler.pilot_study(n_sims_pilot,
                    quantile=quantile,
                    stat_scale='sd',
                    # stat_weight=stat_weights,
                    n_jobs=1,
                    seed=seed_pilot
                    )
'''
scale = np.array([24.41671154, 0.22716134, 1.3784167])
# Sample from posterior
journal = sampler.sample(n_samples_posterior,
                         # use_pilot=True,
                         epsilon=2,
                         stat_scale=scale,
                         # stat_weight=stat_weights,
                         n_jobs=1,
                         # seed=seed_sampler,
                         return_journal=True
                         )
# Save journal
filename = f'bnet_org_test_run.jnl'
journal.save(path + filename)

journal.plot_posterior("g", point_estimate='map', theta_true=5.,)
plt.show()

journal.plot_posterior("eta", point_estimate='map', theta_true=2.,)
plt.show()

# Local linear regression adjustment
journal = sampler.reg_adjust(method="loclinear",
                             transform=True,
                             return_journal=True
                             )

# Save journal
filename = f'bnet_reg_test_run.jnl'
journal.save(path + filename)

journal.plot_posterior("g", point_estimate='map', theta_true=5.,)
plt.show()

journal.plot_posterior("eta", point_estimate='map', theta_true=2.,)
plt.show()
