#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA

# simulator model
simulator = ToyModels.gaussian_unkwown_variance

# summary statistic calculator
summary_calculator = StatisticsCalculator.variance

# distance metric
distance_metric = DistanceMetrics.euclidean


def rejection_ABC(observed_data, priors, n_posterior_samples, n_sims_per_parameter, epsilon):
    """
    Rejection ABC as described by the Pritchard et al. (1999) algorithm.

    This implementation expects predefined function objects:
        1. simulator(*thetas, n_sims_per_parameter)
        2. summary_calculator(data)
        3. distance_metric(sim_sumstat, obs_sumstat)

    Arguments
    ---------
    observed_data : array
        The observed data
    priors : list
        List of priors as scipy.stats objects
    n_posterior_samples : int
        The number of posterior samples to produce
    n_sims_per_parameter : int
        Image filename (with path included) of image with shape (H, W, c) to
        transform
    epsilon : float
        Discrepancy threshold

    Returns
    -------
    samples : list
        ABC posterior samples
    """

    # calculate observed data summary statistic
    obs_sumstat = summary_calculator(observed_data)

    samples = []
    accepted_count = 0

    while accepted_count < n_posterior_samples:
        # draw thetas from priors
        thetas = [theta.rvs() for theta in priors]
        # simulated data given realizations of drawn thetas
        sim_data = simulator(*thetas, N)
        # summary stat of simulated data
        sim_sumstat = summary_calculator(sim_data)
        # calculate discrepancy
        distance = distance_metric(sim_sumstat, obs_sumstat)
        # rejection step
        if distance <= epsilon:
            accepted_count += 1
            samples.append(thetas)

    return samples


def observed_data(N):
    """
    Test problem, single parameter inference

    Arguments
    ---------
    N : int
        Number of observed data samples
    """

    # observed data
    groundtruth = 2.0
    likelihood = stats.norm(loc=0., scale=np.sqrt(groundtruth))
    obs_data = likelihood.rvs(size=N)

    # priors
    alpha = 60
    beta = 130
    sigma2 = stats.InvGamma(k=alpha, loc=0, beta=beta)
    priors = [sigma2]

    # inference config
    n_posterior_samples = 1000
    n_simulator_samples_per_parameter = N
    epsilon = 0.1

    return obs_data, priors


def plain():
    pass


def N_obs_data_analysis():
    pass


def priors_analysis():
    # priors

    # Conjugate, informative prior
    sigma2 = stats.InvGamma(k=60, loc=0, beta=130)
    priors = [sigma2]


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence


# Should be normalized though
values1 = np.asarray([1.346112, 1.337432, 1.246655])
values2 = np.asarray([1.033836, 1.082015, 1.117323])

# Note slight difference in the final result compared to Dawny33
print KL(values1, values2)  # 0.775278939433


# PCA


data = X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

#pca = PCA(n_components='mle', whiten=False)
#data = pca.fit_transform(data)

pca = PCA().fit(data)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

if __name__ == "__main__":
    print(2)
