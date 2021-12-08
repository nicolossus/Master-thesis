import matplotlib.pyplot as plt
import numpy as np
import pylfi
import scipy.stats as stats

# Observed data
mu_true = 163
sigma_true = 15
likelihood = stats.norm(loc=mu_true, scale=sigma_true)
obs_data = likelihood.rvs(size=1000)


# Simulator model
def simulator(mu, sigma, size=1000):
    y_sim = stats.norm(loc=mu, scale=sigma).rvs(size=size)
    return y_sim


# Summary statistics calculator
def stat_calc(y):
    sum_stat = [np.mean(y), np.std(y)]
    return sum_stat


# Priors
mu_prior = pylfi.Prior('norm',
                       loc=165,
                       scale=2,
                       name='mu',
                       tex='$\mu$'
                       )

sigma_prior = pylfi.Prior('uniform',
                          loc=12,
                          scale=7,
                          name='sigma',
                          tex='$\sigma$'
                          )


priors = [mu_prior, sigma_prior]

# Initialize sampler
sampler = pylfi.RejABC(obs_data,
                       simulator,
                       stat_calc,
                       priors,
                       log=True
                       )

# Pilot study
nsims = 1000
sampler.pilot_study(nsims,
                    quantile=0.2,
                    stat_scale="sd",
                    stat_weight=1,
                    n_jobs=4,
                    seed=4
                    )

# Sample posterior
nsamples = 3000
sampler.sample(nsamples,
               use_pilot=True,
               n_jobs=4,
               seed=42
               )

# Local linear regression adjustment
journal = sampler.reg_adjust(method="loclinear",
                             kernel="epkov",
                             transform=True,
                             return_journal=True
                             )


# Save journal
filename = 'my_journal.jnl'
journal.save(filename)

# Load journal
journal = pylfi.Journal.load(filename)

# pandas dataframe with posterior samples
df = journal.df

# Plot posteriors
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 3), tight_layout=True)
journal.plot_posterior('mu',
                       hdi_prob=0.95,
                       point_estimate='map',
                       theta_true=mu_true,
                       ax=axes[0]
                       )

journal.plot_posterior('sigma',
                       hdi_prob=0.95,
                       point_estimate='map',
                       theta_true=sigma_true,
                       ax=axes[1]
                       )

plt.show()
