#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from journal import Journal

'''
change implementation to something like this:

sbi takes any function as simulator. Thus, sbi also has the flexibility to use
simulators that utilize external packages, e.g., Brian (http://briansimulator.org/),
nest (https://www.nest-simulator.org/), or NEURON (https://neuron.yale.edu/neuron/).
External simulators do not even need to be Python-based as long as they store
simulation outputs in a format that can be read from Python. All that is necessary
is to wrap your external simulator of choice into a Python callable that takes a
parameter set and outputs a set of summary statistics we want to fit the parameters to

* simulator must return summary statistics
* then in init, only simulator is needed
* change distance metric to keyword and allow for custom callable
* remove n_simulator_samples_per_parameter from sample method 
'''


class RejectionABC:

    def __init__(self, simulator, summary_calculator, distance_metric):
        """
        simulator : callable
            simulator model
        summary_calculator : callable
            summary statistics calculator
        distance_metric : callable
            discrepancy measure
        """

        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

    def sample(self, observed_data, priors, n_posterior_samples, n_simulator_samples_per_parameter, epsilon):
        """
        Pritchard et al. (1999) algorithm

        n_samples: integer
            Number of samples to generate
        """

        _inference_scheme = "Rejection ABC"
        N = n_simulator_samples_per_parameter

        obs_sumstat = self._summary_calc(
            observed_data)  # observed data summary statistic

        journal = Journal()  # journal instance
        journal._start_journal()

        journal._add_config(self._simulator, self._summary_calc, self._distance_metric,
                            _inference_scheme, n_posterior_samples, n_simulator_samples_per_parameter, epsilon)
        journal._add_parameter_names(priors)

        number_of_simulations = 0
        accepted_count = 0

        while accepted_count < n_posterior_samples:
            number_of_simulations += 1
            # draw thetas from priors
            thetas = [theta.rvs() for theta in priors]
            # simulated data given realizations of drawn thetas
            sim_data = self._simulator(*thetas, N)
            # summary stat of simulated data
            sim_sumstat = self._summary_calc(sim_data)
            # calculate distance
            distance = self._distance_metric(sim_sumstat, obs_sumstat)

            if distance <= epsilon:
                accepted_count += 1
                journal._add_accepted_parameters(thetas)
                journal._add_distance(distance)
                journal._add_sumstat(sim_sumstat)

        journal._add_sampler_summary(number_of_simulations, accepted_count)

        return journal


class MCMCABC:

    def __init__(self, simulator, summary_calculator, distance_metric):

        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

    def sample(self):
        pass


class SMCABC:

    def __init__(self, simulator, summary_calculator, distance_metric):

        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

    def sample(self):
        pass


class PMCABC:

    def __init__(self, simulator, summary_calculator, distance_metric):

        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

    def sample(self):
        pass
