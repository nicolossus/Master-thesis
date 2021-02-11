#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np


class Journal:

    def __init__(self):

        self.accepted_parameters = {}
        self.parameter_names = []
        self.distances = []

        self.configuration = {}
        self.sampler_summary = {}

        self._journal_started = False

    def _start_journal(self):
        self._journal_started = True

    def _check_journal_status(self):
        if not self._journal_started:
            msg = "Journal unavailable; run an inference scheme first"
            raise ValueError(msg)

    def _add_config(self, simulator, summary_calc, distance_metric, inference_scheme, n_posterior_samples, n_simulator_samples_per_parameter, epsilon):
        """
        docs
        """

        self.configuration["Simulator model"] = simulator.__name__
        self.configuration["Summary calculator"] = summary_calc.__name__
        self.configuration["Distance metric"] = summary_calc.__name__
        self.configuration["Inference scheme"] = inference_scheme
        self.configuration["Number of posterior samples"] = n_posterior_samples
        self.configuration["Number of simulator samples per parameter"] = n_simulator_samples_per_parameter
        self.configuration["Epsilon"] = epsilon

    def _add_parameter_names(self, priors):
        for parameter in priors:
            self.parameter_names.append(parameter.name)
            self.accepted_parameters[parameter.name] = []

    def _add_accepted_parameters(self, thetas):
        """
        docs
        """

        for parameter_name, theta in zip(self.parameter_names, thetas):
            self.accepted_parameters[parameter_name].append(theta)

    def _add_distance(self, distance):
        """
        docs
        """

        self.distances.append(distance)

    def _add_sampler_summary(self, number_of_simulations, accepted_count):
        """
        docs
        """

        accept_ratio = accepted_count / number_of_simulations
        # number of parameters estimated
        self.sampler_summary["Number of simulations"] = number_of_simulations
        self.sampler_summary["Number of accepted simulations"] = accepted_count
        self.sampler_summary["Acceptance ratio"] = accept_ratio
        # posterior means
        # uncertainty

    @property
    def get_accepted_parameters(self):
        """
        docs
        """
        return self.accepted_parameters

    @property
    def get_distances(self):
        """
        docs
        """

        return self.distances

    @property
    def get_number_of_simulations(self):
        self._check_journal_status()
        return self.sampler_summary["Number of simulations"]

    @property
    def get_number_of_accepted_simulations(self):
        self._check_journal_status()
        return self.sampler_summary["Number of accepted simulations"]

    @property
    def get_acceptance_ratio(self):
        self._check_journal_status()
        return self.sampler_summary["Acceptance ratio"]

    def save(self, filename):
        """
        Stores the journal to disk.

        Parameters
        ----------
        filename: string
            the location of the file to store the current object to.
        """

        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)

    def load(self):
        pass

    def posterior_kde(self, kernel="gaussian"):
        pass
