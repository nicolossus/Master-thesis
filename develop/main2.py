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
from numpy.random import default_rng
from priors import *
from simulators import *
from summary_statistics import *

# observed height data
# https://rstudio-pubs-static.s3.amazonaws.com/349638_84f7c91e43c54eccb346a1f550736d01.html
rng = default_rng()
N = 150

mu_true = 163
sigma_true = 15
true_parameter_values = [mu_true, sigma_true]
likelihood = stats.norm(loc=mu_true, scale=sigma_true)
data = likelihood.rvs(size=N)

sigma_noise = 0.1
noise = rng.normal(0, sigma_noise, N)

#obs_data = data + noise
obs_data = data

# simulator model
gaussian_unkwown_variance = ToyModels.gaussian_model

# summary statistic calculator
variance = StatisticsCalculator.mean

# distance metric
euclidean = DistanceMetrics.euclidean

# initialize sampler
sampler = RejectionABC(simulator=gaussian_unkwown_variance,
                       summary_calculator=variance, distance_metric=euclidean)

# priors
mu = Normal(165, 20, name="mu", tex="$\mu$")
sigma = Uniform(5, 30, name="sigma", tex="$\sigma$")
priors = [mu, sigma]

# inference config
n_posterior_samples = 100
n_simulator_samples_per_parameter = N
epsilon = 0.1

# run inference
journal = sampler.sample(obs_data, priors, n_posterior_samples,
                         n_simulator_samples_per_parameter, epsilon)

# journal
samples_mu = journal.get_accepted_parameters["mu"]
samples_sigma = journal.get_accepted_parameters["sigma"]

#samples_mu, samples_sigma = journal.get_params

#*samples, = journal.get_params
*samples, = journal._get_params_as_arrays()

for sample in samples:
    print(np.mean(sample))

journal.histplot(true_parameter_values=true_parameter_values)
# plt.show()

#kernel_mu = stats.gaussian_kde(np.array(samples_mu).T[0])
#kernel_sigma = stats.gaussian_kde(np.array(samples_sigma).T[0])

'''
samples_mu = np.asarray(samples_mu, float)
if samples_mu.ndim > 1:
    samples_mu = samples_mu.squeeze()

samples_sigma = np.asarray(samples_sigma, float)
if samples_sigma.ndim > 1:
    samples_sigma = samples_sigma.squeeze()

print(np.mean(samples_mu))
print(np.mean(samples_sigma))

fig, ax = plt.subplots(1, 2, tight_layout=True)
# mu
ax[0].hist(samples_mu, density=True, histtype='bar',
           edgecolor=None, color='steelblue', alpha=0.5, label="mu samples")
# sigma
ax[1].hist(samples_sigma, density=True, histtype='bar',
           edgecolor=None, color='steelblue', alpha=0.5, label="sigma samples")
plt.legend()
plt.show()

# plot observed data
(dmin, dmax) = (100, 230)
lh_data = likelihood.pdf(obs_data)
x_arr = np.arange(dmin, dmax, (dmax - dmin) / 100.)
f_arr = likelihood.pdf(x_arr)

fig = plt.figure()
plt.plot(x_arr, mu.pdf(x_arr))
plt.show()
'''

"""
fig = plt.figure(figsize=(10, 6))
plt.xlim([dmin, dmax])
plt.plot(x_arr, f_arr, lw=2.5, label="Likelihood")
markerline, stemlines, baseline = plt.stem(
    obs_data, lh_data, linefmt='-k', markerfmt='k.', label="Observed Data")
# markerline.set_markerfacecolor('none')
plt.setp(stemlines, alpha=0.4)
baseline.set_visible(False)
plt.title("Likelihood & Observed Data")
plt.xlabel("Height [cm]")
plt.legend()
plt.ylim(bottom=0.)
plt.show()

fig, ax = plt.subplots(1, 2)
# mu
ax[0].hist(samples_mu, density=True, histtype='bar',
           edgecolor=None, color='steelblue', alpha=0.5, label="mu samples")
ax[0].plot(x, kernel_mu.evaluate(x_arr), label="approximate posterior")
# sigma
ax[1].hist(samples_sigma, density=True, histtype='bar',
           edgecolor=None, color='steelblue', alpha=0.5, label="sigma samples")
ax[1].plot(x, kernel_sigma.evaluate(x_arr), label="approximate posterior")
plt.show()
"""
