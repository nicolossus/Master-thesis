#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from distances import *
from inferences import *
from journal import *
from priors import *
from simulators import *
from summary_statistics import *

# observed data
N = 1000

groundtruth = 2.0
likelihood = stats.norm(loc=0., scale=np.sqrt(groundtruth))
obs_data = likelihood.rvs(size=N)

# simulator model
gaussian_unkwown_variance = ToyModels.gaussian_unkwown_variance

# summary statistic calculator
variance = StatisticsCalculator.variance

# distance metric
euclidean = DistanceMetrics.euclidean

# initialize sampler
sampler = RejectionABC(simulator=gaussian_unkwown_variance,
                       summary_calculator=variance, distance_metric=euclidean)

# priors
#sigma2 = Uniform(name="sigma2")

alpha = 60
beta = 130
sigma2 = InvGamma(k=alpha, loc=0, beta=beta, name="sigma2", tex="$\sigma^2$")
priors = [sigma2]

# inference config
n_posterior_samples = 1000
n_simulator_samples_per_parameter = N
epsilon = 0.1

# run inference
journal = sampler.sample(obs_data, priors, n_posterior_samples,
                         n_simulator_samples_per_parameter, epsilon)

# check journal
'''
print(journal.configuration)
print(journal.get_number_of_simulations)
print(journal.get_number_of_accepted_simulations)
print(journal.get_acceptance_ratio)
# print(journal.get_accepted_parameters)
print()
'''

# kernel density estimation of the approximate posterior
samples = journal.get_accepted_parameters["sigma2"]
kernel = stats.gaussian_kde(np.array(samples).T[0])

fraction_accepted = journal.get_acceptance_ratio

a = 1
b = 4
x = np.arange(a, b, 0.01)

point_estimate = np.mean(samples)


alphaprime = alpha + N / 2
betaprime = beta + 0.5 * np.sum(obs_data**2)
posterior = stats.invgamma(alphaprime, loc=0, scale=betaprime)

print(f"groundtruth = {groundtruth}, point estimate = {point_estimate:.3f}")
journal.histplot(true_parameter_values=[2])
'''
print(np.mean(kernel.evaluate(x), axis=0))

samples = np.array(samples).reshape(-1, )

fig, ax = plt.subplots(1, 2)
sns.distplot(samples, label='samples', ax=ax[0])
#plt.plot(x, posterior.pdf(x), color='red', lw=1.5, label="true posterior")
#plt.plot(x, kernel.evaluate(x), label="approximate posterior")
# plt.legend()
# plt.show()

#fig = plt.figure()
ax[1].hist(samples, density=True, histtype='bar',
           edgecolor=None, color='steelblue', alpha=0.5, label="samples")
ax[1].plot(x, kernel.evaluate(x), label="approximate posterior")
#plt.plot(x, posterior.pdf(x) / posterior.pdf(x).max(), label="true posterior")
plt.xlim(1.5, 2.5)
plt.show()
'''

'''
fig = plt.figure()
plt.hist(samples, density=True, histtype='stepfilled',
         edgecolor='black', color='steelblue', alpha=0.5, bins=50, label="samples")
plt.plot(x, posterior.pdf(x), color='red', lw=1.5, label="true posterior")
#plt.plot(x, kernel.evaluate(x), label="approximate posterior")
plt.xlim(1, 3)
plt.legend()
plt.show()

fig = plt.figure()
plt.plot([groundtruth, groundtruth], [0, 1.2],
         linestyle='--', color='black', label="groundtruth")
plt.plot([point_estimate, point_estimate], [0, 1.2],
         linestyle='--', color='red', label="point estimate")
plt.plot(x, posterior.pdf(x) / posterior.pdf(x).max(), label="true posterior")
plt.plot(x, kernel.evaluate(x) / kernel.evaluate(x).max(),
         label="approximate posterior")
plt.xlabel("$\sigma^2$")
plt.title("$\\varepsilon=" + str(epsilon) +
          "$, acceptance ratio=" + str(round(fraction_accepted, 3)))
plt.xlim(1, 3)
plt.legend()
plt.show()
'''

'''
fig = plt.figure()
plt.plot([groundtruth, groundtruth], [0, 1.2],
         linestyle='--', color='black', label="groundtruth")
plt.plot([point_estimate, point_estimate], [0, 1.2],
         linestyle='--', color='red', label="point estimate")
plt.plot(x, kernel.evaluate(x) / kernel.evaluate(x).max(),
         label="approximate posterior")
plt.xlabel("$\sigma^2$")
plt.title("$\\varepsilon=" + str(epsilon) +
          "$, acceptance ratio=" + str(round(fraction_accepted, 3)))
plt.legend()
plt.show()
'''
