#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from distances import *
from inferences import *
from journal import *
from priors import *
from simulators import *
from summary_statistics import *

# observed data
N = 100

groundtruth = 0.5
likelihood = stats.norm(loc=0., scale=np.sqrt(groundtruth))
obs_data = likelihood.rvs(size=N)

# simulator model
gaussian_unkwown_variance = ToyModels.gaussian_unkwown_variance

# summary statistic calculator
variance = StatisticsCalculator.variance

# distance metric
euclidean = DistanceMetrics.euclidean

# initialize sampler
sampler = ABCSampler(observed_data=obs_data, simulator=gaussian_unkwown_variance,
                     summary_calculator=variance, distance_metric=euclidean)

# priors
sigma2 = Uniform(name="sigma^2")
priors = [sigma2]

# inference config
n_posterior_samples = 5
n_simulator_samples_per_parameter = N
epsilon = 0.5

# run inference
journal = sampler.rejection_abc(
    priors, n_posterior_samples, n_simulator_samples_per_parameter, epsilon)

# check journal
print(journal.configuration)
print(journal.get_number_of_simulations)
print(journal.get_number_of_accepted_simulations)
print(journal.get_acceptance_ratio)
print(journal.get_accepted_parameters)
