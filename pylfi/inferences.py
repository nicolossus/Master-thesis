#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from journal import Journal


class ABCSampler:

    def __init__(self, observed_data, simulator, summary_calculator, distance_metric):

        self._obs_data = observed_data           # observed data
        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

        self._obs_sumstat = self._summary_calc(
            self._obs_data)  # observed data summary statistic

    def rejection_abc(self, priors, n_posterior_samples, n_simulator_samples_per_parameter, epsilon):
        """
        n_samples: integer
            Number of samples to generate
        """

        _inference_scheme = "Rejection ABC"
        N = n_simulator_samples_per_parameter

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
            distance = self._distance_metric(sim_sumstat, self._obs_sumstat)

            if distance <= epsilon:
                accepted_count += 1
                journal._add_accepted_parameters(thetas)
                journal._add_distance(distance)

        journal._add_sampler_summary(number_of_simulations, accepted_count)

        return journal

    def mcmc_abc(self):
        pass

    def smc_abc(self):
        pass

    def pmc_abc(self):
        pass
