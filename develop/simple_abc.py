#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, invgamma, norm

# Set fontsizes in figures
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large',
          'legend.handlelength': 2}
plt.rcParams.update(params)

np.random.seed(42)


def distance(sim, data):
    return np.sqrt(np.sum((sim - data) * (sim - data)))


def rejection_sampler():

    n_sample = 100000
    prior = np.random.uniform(low=0, high=1, size=n_sample)
    pass


def rejection_abc(obs_data, prior, simulator, summary_stat, threshold=0.5, Nsims=1000):
    """
    """
    samples = []
    data_ss = summary_stat(obs_data)  # observed data summary statistic

    for trial in range(Nsims):
        var_trial = prior.rvs(size=1)     # draw from prior
        sim_trial = simulator(var_trial)  # call simulator
        sim_ss = summary_stat(sim_trial)  # simulated data summary statistic
        # keep or discard simulation
        if distance(sim_ss, data_ss) < threshold:
            samples.append(var_trial)

    accept_ratio = float(len(samples)) / Nsims
    # kernel density estimation of the approximate posterior
    kernel = gaussian_kde(np.array(samples).T[0])
    return kernel, accept_ratio


def mod_rejection_abc(obs_data, prior, simulator, summary_stat, threshold=0.5, Nsims=1000):
    """
    """
    samples = []
    data_ss = summary_stat(obs_data)  # observed data summary statistic
    draw_posterior = False

    for trial in range(Nsims):
        if draw_posterior:
            # kernel density estimation of the approximate posterior
            posterior_approx = gaussian_kde(np.array(samples).T[0])
            var_trial = posterior_approx.resample(size=1)[0]
        else:
            var_trial = prior.rvs(size=1)     # draw from prior

        sim_trial = simulator(var_trial)  # call simulator
        sim_ss = summary_stat(sim_trial)  # simulated data summary statistic

        # keep or discard simulation
        if distance(sim_ss, data_ss) < threshold:
            samples.append(var_trial)
            if not draw_posterior and len(samples) > 50:
                draw_posterior = True

    accept_ratio = float(len(samples)) / Nsims
    # kernel density estimation of the approximate posterior
    kernel = gaussian_kde(np.array(samples).T[0])
    return kernel, accept_ratio


if __name__ == "__main__":
    # data (likelihood)
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
    Nsims = 5000
    threshold = 0.2

    kernel1, accept_ratio1 = rejection_abc(data, prior, simulator,
                                           summary_stat, threshold, Nsims)

    kernel2, accept_ratio2 = mod_rejection_abc(data, prior, simulator,
                                               summary_stat, threshold, Nsims)

    # produce a plot
    plt.figure(figsize=(10, 6))
    plt.xlim([a, b])
    plt.xlabel("$\sigma^2$")
    plt.ylim([0, 1.2])
    plt.plot([groundtruth, groundtruth], [0, 1.2],
             linestyle='--', color='black', label="groundtruth")
    plt.plot(x, prior.pdf(x) / prior.pdf(x).max(), '--', label="prior")
    plt.plot(x, posterior.pdf(x) / posterior.pdf(x).max(),
             label="true posterior")
    plt.plot(x, kernel1.evaluate(x) / kernel1.evaluate(x).max(),
             label="approx. posterior 1 (prior draw)")
    plt.plot(x, kernel2.evaluate(x) / kernel2.evaluate(x).max(),
             label="approx. posterior 2 (posterior draw)")
    plt.title("$\\varepsilon=" + str(threshold) +
              "$, accept ratio 1=" + str(round(accept_ratio1, 4)) + ", accept ratio 2=" + str(round(accept_ratio2, 4)))
    plt.legend()
    plt.show()


'''
for epsilon in [0.5, 0.4, 0.3, 0.2, 0.1]:

    samples = rejection_abc(epsilon, Nsims)

    fraction_accepted = float(len(samples)) / Nsims

    # kernel density estimation of the approximate posterior
    kernel = gaussian_kde(samples)

    # produce a plot
    plt.figure(figsize=(10, 6))
    plt.xlim([a, b])
    plt.xlabel("$\sigma^2$")
    plt.ylim([0, 1.2])
    plt.plot([groundtruth, groundtruth], [0, 1.2],
             linestyle='--', color='black', label="groundtruth")
    plt.plot(x, prior.pdf(x) / prior.pdf(x).max(), label="prior")
    plt.plot(x, posterior.pdf(x) / posterior.pdf(x).max(),
             label="true posterior")
    plt.plot(x, kernel.evaluate(x) / kernel.evaluate(x).max(),
             label="approximate posterior")
    plt.title("$\\varepsilon=" + str(epsilon) +
              "$, acceptance ratio=" + str(fraction_accepted))
    plt.legend()
    plt.show()
'''
