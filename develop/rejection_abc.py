#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, invgamma, norm

# wrap numpy
'''
Problem:

Traceback (most recent call last):
  File "/Users/nicolai/github/Master-thesis/src/abc/rejection_abc.py", line 201, in <module>
    kernel1, accept_ratio1 = rejection_abc(
  File "/Users/nicolai/github/Master-thesis/src/abc/rejection_abc.py", line 157, in rejection_abc
    kernel = gaussian_kde(np.array(samples).T[0])
  File "/Users/nicolai/opt/anaconda3/lib/python3.8/site-packages/scipy/stats/kde.py", line 206, in __init__
    self.set_bandwidth(bw_method=bw_method)
  File "/Users/nicolai/opt/anaconda3/lib/python3.8/site-packages/scipy/stats/kde.py", line 556, in set_bandwidth
    self._compute_covariance()
  File "/Users/nicolai/opt/anaconda3/lib/python3.8/site-packages/scipy/stats/kde.py", line 568, in _compute_covariance
    self._data_inv_cov = linalg.inv(self._data_covariance)
  File "/Users/nicolai/opt/anaconda3/lib/python3.8/site-packages/scipy/linalg/basic.py", line 977, in inv
    raise LinAlgError("singular matrix")
numpy.linalg.LinAlgError: singular matrix

https://stats.stackexchange.com/questions/89754/statsmodels-error-in-kde-on-a-list-of-repeated-values

###

theta = prior.rvs(
    size=1, random_state=np.random.RandomState(seed=rng_seed))

#####

def beta(a, b, size=None):
    return np.random.default_rng().beta(a, b, size)


def binomial(n, p, size=None):
    return np.random.default_rng().binomial(n, p, size)


def chisquare(df, size=None):
    return np.random.default_rng().chisquare(df, size)


def dirichlet(alpha, size=None):
    return np.random.default_rng().dirichlet(alpha, size)


def exponential(scale=1.0, size=None):
    return np.random.default_rng().exponential(scale, size)


def rejection_sampler(method="rejection_abc"):
    samples = []
    for trial in range(N_sims):
        if method == "rejection_abc":

    pass


def rejection_abc_one_theta(prior, simulator, data, ss_stat, N_sims=1000, distance="euclidean", threshold=0.5):
    samples = []
    data_ss = ss_stat(data)                 # data sufficient summary statistic

    for trial in range(N_sims):
        theta_trial = prior.rvs(size=1)     # draw from prior
        sim_trial = simulator(theta_trial)  # call simulator
        sim_ss = ss_stat(sim_trial)         # sim. sufficient summary statistic
        reject_bool = distance(sim_ss, data_ss, threshold)

        # keep or discard simulation
        if reject_bool:
            samples.append(var_trial)

    return np.array(samples).T[0]


class Prior:
    def __init__():
        pass


class ABC:
    def __init__(self, data, simulator):
        self._data_ss = None

    def ss_stat(self):
        pass

    def rejection_abc(self):


def rejection_abc(epsilon, Nsims):
    samples = []
    for trial in range(Nsims):
        # draw from prior
        var_trial = prior.rvs(size=1)

        # call simulator
        sim_trial = simulator(var_trial)

        # summary statistic
        sim_ss = sufficient_summary_stat(sim_trial)

        # distance less than threshold?
        reject_bool = distance(sim_ss, data_ss) < epsilon

        # keep or discard simulation
        if reject_bool:
            samples.append(var_trial)

    return np.array(samples).T[0]
'''

######
######


def distance(sim, data):
    return np.sqrt(np.sum((sim - data) * (sim - data)))


def rejection_sampler(obs_data, prior, simulator, summary_stat, epsilon=0.5, Nsims=1000):
    # observed data summary statistic
    obs_sumstat = summary_stat(obs_data)
    # draw thetas from prior
    thetas = prior.rvs(size=Nsims)
    # simulated data given a realization of drawn theta
    sim_data = [simulator(theta) for theta in thetas]
    # summary stat of simulated data
    sim_sumstats = [summary_stat(sim) for sim in sim_data]
    # rejection sampler
    samples = [[thetas[i]] for i, sim_sumstat in enumerate(
        sim_sumstats) if distance(sim_sumstat, obs_sumstat) < epsilon]
    # compute acceptance ratio
    accept_ratio = float(len(samples)) / Nsims
    # kernel density estimation of the approximate posterior
    kernel = gaussian_kde(np.array(samples).T[0])

    return kernel, accept_ratio


def rejection_abc(obs_data, prior, simulator, summary_stat, epsilon=0.5, Nsims=1000):

    samples = []
    obs_sumstat = summary_stat(obs_data)  # observed data summary statistic

    for i in range(Nsims):
        # draw theta from prior
        theta = prior.rvs(size=1)
        # simulated data given a realization of drawn theta
        sim_data = simulator(theta)
        # simulated data summary statistic
        sim_sumstat = summary_stat(sim_data)
        # keep or discard simulation
        if distance(sim_sumstat, obs_sumstat) < epsilon:
            samples.append(theta)

    # compute acceptance ratio
    accept_ratio = float(len(samples)) / Nsims
    # kernel density estimation of the approximate posterior
    kernel = gaussian_kde(np.array(samples).T[0])

    return kernel, accept_ratio


if __name__ == "__main__":
    # data (likelihood)
    rng_seed = 42
    # np.random.seed(rng_seed)
    Nsamp = 500

    groundtruth = 2.
    likelihood = norm(loc=0., scale=np.sqrt(groundtruth))

    (dmin, dmax) = (-5, 5)
    data = likelihood.rvs(size=Nsamp)
    lh_data = likelihood.pdf(data)
    x_arr = np.arange(dmin, dmax, (dmax - dmin) / 100.)
    f_arr = likelihood.pdf(x_arr)

    # prior and posterior
    alpha = 60
    beta = 130
    prior = invgamma(alpha, loc=0, scale=beta)

    alphaprime = alpha + Nsamp / 2
    betaprime = beta + 0.5 * np.sum(data**2)
    posterior = invgamma(alphaprime, loc=0, scale=betaprime)

    a = 1
    b = 4
    x = np.arange(a, b, 0.01)

    # define simulator model
    def simulator(var):
        return norm(loc=0., scale=np.sqrt(var)).rvs(size=Nsamp)

    # define summary statistic (this particular is sufficient for normal model)
    def summary_stat(data):
        return np.var(data)

    # run rejection ABC sampler
    Nsims = 100
    epsilon = 0.5

    kernel1, accept_ratio1 = rejection_abc(
        data, prior, simulator, summary_stat, epsilon, Nsims, rng_seed=42)

    kernel2, accept_ratio2 = rejection_sampler(
        data, prior, simulator, summary_stat, epsilon, Nsims, rng_seed=42)

    print(accept_ratio1)
    print(accept_ratio2)
