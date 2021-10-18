import os

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import pandas as pd
import pylfi
import scipy.stats as stats
import seaborn as sns

# Set plot style
sns.set(context="paper", style='darkgrid', rc={"axes.facecolor": "0.96"})


def sem(a, axis=0):
    """Standard error of the mean (SEM)"""
    return np.std(a) / np.sqrt(a.shape[axis])


def rmse(y_true, y_pred):
    """Root mean square error (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def rmspe(y_true, y_pred):
    """Root mean square percentage error (RMSPE)"""
    rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))) * 100
    return rmspe


gbarK0 = 36.    # ground truth
gbarNa0 = 120.  # ground truth
gbarK_true = np.ones(1000) * gbarK0
gbarNa_true = np.ones(1000) * gbarNa0

path = 'data/'

# iterators
quantile_lst = [0.1, 0.3, 0.5, 0.7, 0.9]
trials = 10
N = len(quantile_lst)

# original posterior samples
gbarK_org_rmspe_mean_norm = np.zeros(N)
gbarK_org_rmspe_sem_norm = np.zeros(N)
gbarNa_org_rmspe_mean_norm = np.zeros(N)
gbarNa_org_rmspe_sem_norm = np.zeros(N)

# regression adjusted posterior samples
gbarK_reg_rmspe_mean_norm = np.zeros(N)
gbarK_reg_rmspe_sem_norm = np.zeros(N)
gbarNa_reg_rmspe_mean_norm = np.zeros(N)
gbarNa_reg_rmspe_sem_norm = np.zeros(N)

for i, quantile in enumerate(quantile_lst):
    quantile_str = str(quantile).replace('.', '_')

    # original posterior samples
    trial_gbarK_rmspe_org_norm = np.zeros(trials)
    trial_gbarNa_rmspe_org_norm = np.zeros(trials)

    # regression adjusted posterior samples
    trial_gbarK_rmspe_reg_norm = np.zeros(trials)
    trial_gbarNa_rmspe_reg_norm = np.zeros(trials)

    for trial in range(trials):
        # Original posterior samples

        # normal prior
        filename = f'hh_rej_normal_org_quantile_{quantile_str}_run_{trial}.jnl'
        journal = pylfi.Journal.load(path + filename)
        df = journal.df

        trial_gbarK_rmspe_org_norm[trial] = rmspe(
            gbarK_true, df["gbarK"].to_numpy())
        trial_gbarNa_rmspe_org_norm[trial] = rmspe(
            gbarNa_true, df["gbarNa"].to_numpy())

        # Regression adjusted posterior samples

        # normal prior
        filename = f'hh_rej_normal_reg_quantile_{quantile_str}_run_{trial}.jnl'
        journal = pylfi.Journal.load(path + filename)
        df = journal.df
        trial_gbarK_rmspe_reg_norm[trial] = rmspe(
            gbarK_true, df["gbarK"].to_numpy())
        trial_gbarNa_rmspe_reg_norm[trial] = rmspe(
            gbarNa_true, df["gbarNa"].to_numpy())

    # Mean and SEM of RMSPE samples

    # Original posterior

    # normal prior
    gbarK_org_rmspe_mean_norm[i] = np.mean(trial_gbarK_rmspe_org_norm)
    gbarK_org_rmspe_sem_norm[i] = sem(trial_gbarK_rmspe_org_norm)
    gbarNa_org_rmspe_mean_norm[i] = np.mean(trial_gbarNa_rmspe_org_norm)
    gbarNa_org_rmspe_sem_norm[i] = sem(trial_gbarNa_rmspe_org_norm)

    # Adjusted posterior
    # normal prior
    gbarK_reg_rmspe_mean_norm[i] = np.mean(trial_gbarK_rmspe_reg_norm)
    gbarK_reg_rmspe_sem_norm[i] = sem(trial_gbarK_rmspe_reg_norm)
    gbarNa_reg_rmspe_mean_norm[i] = np.mean(trial_gbarNa_rmspe_reg_norm)
    gbarNa_reg_rmspe_sem_norm[i] = sem(trial_gbarNa_rmspe_reg_norm)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(
    4, 2), tight_layout=True, dpi=300, sharex=True)

# gbarK
ax1 = axes[0]
# informative prior
ax1.errorbar(quantile_lst,
             gbarK_org_rmspe_mean_norm,
             yerr=gbarK_org_rmspe_sem_norm,
             fmt='-o',
             color='C0',
             ecolor='k',
             markersize=3.5,
             lw=1.0,
             elinewidth=1.5,
             capsize=3,
             )

ax1.errorbar(quantile_lst,
             gbarK_reg_rmspe_mean_norm,
             yerr=gbarK_reg_rmspe_sem_norm,
             fmt='-o',
             color='C1',
             ecolor='k',
             markersize=3.5,
             lw=1.0,
             elinewidth=1.5,
             capsize=3,
             )
ax1.set(ylabel="RMSPE (\%)",
        title=r'$\bar{g}_\mathrm{K}$'
        )

# gbarNa
ax2 = axes[1]
# informative prior
ax2.errorbar(quantile_lst,
             gbarNa_org_rmspe_mean_norm,
             yerr=gbarNa_org_rmspe_sem_norm,
             fmt='-o',
             color='C0',
             ecolor='k',
             markersize=3.5,
             lw=1.0,
             elinewidth=1.5,
             capsize=3,
             )

ax2.errorbar(quantile_lst,
             gbarNa_reg_rmspe_mean_norm,
             yerr=gbarNa_reg_rmspe_sem_norm,
             fmt='-o',
             color='C1',
             ecolor='k',
             markersize=3.5,
             lw=1.0,
             elinewidth=1.5,
             capsize=3,
             )

ax2.set(xticks=quantile_lst,
        xlabel=r'$p_{\epsilon}$',
        ylabel="RMSPE (\%)",
        title=r'$\bar{g}_\mathrm{Na}$'
        )

plt.show()
