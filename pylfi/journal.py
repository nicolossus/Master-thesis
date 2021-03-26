#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# TODO: store in pandas dataframe


class Journal:

    def __init__(self):

        self.accepted_parameters = {}
        self.parameter_names = []
        self.parameter_names_tex = []
        self.labels = []
        self.distances = []
        self.sumstats = []
        self._n_parameters = 0

        self.configuration = {}
        self.sampler_summary = {}

        self._journal_started = False

    def _start_journal(self):
        self._journal_started = True

    def _check_journal_status(self):
        if not self._journal_started:
            msg = "Journal unavailable; run an inference scheme first"
            raise ValueError(msg)

    def _check_true_parameter_values(self, true_parameter_values):
        if not isinstance(true_parameter_values, list):
            msg = "True parameter values must be provided in a list"
            raise ValueError(msg)
        if self._n_parameters != len(true_parameter_values):
            msg = "The number of true parameter values in list must equal the number of inferred parameters."
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
            name = parameter.name
            tex = parameter.tex
            self.parameter_names.append(name)
            self.accepted_parameters[name] = []
            self.parameter_names_tex.append(tex)
            self._n_parameters += 1
            if tex is None:
                self.labels.append(name)
            else:
                self.labels.append(tex)

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

    def _add_sumstat(self, sumstat):
        """
        docs
        """

        self.sumstats.append(sumstat)

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

    def _get_params_as_arrays(self):
        """
        Transform data of accepted parameters to 1D arrays
        """

        samples = self.get_accepted_parameters
        if len(self.parameter_names) > 1:
            params = (np.asarray(samples[name], float).squeeze() if np.asarray(
                samples[name], float).ndim > 1 else np.asarray(samples[name], float) for name in self.parameter_names)
        else:
            samples = np.asarray(samples[self.parameter_names[0]], float)
            params = samples.squeeze() if samples.ndim > 1 else samples
        return params

    def _point_estimates(self):
        """
        Calculate point estimate of inferred parameters
        """
        *samples, = self._get_params_as_arrays()
        if self._n_parameters == 1:
            point_estimates = [np.mean(samples)]
        else:
            point_estimates = [np.mean(sample) for sample in samples]
        return point_estimates

    @property
    def get_distances(self):
        """
        docs
        """
        self._check_journal_status()
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

    def _samples(self, name):
        pass

    def _kde(self):
        pass

    def _square_root_rule(x):
        """
        Calculate number of histogram bins using the Square-root Rule.
        """
        x = np.asarray(data)
        n = x.size
        k = int(np.ceil(np.sqrt(n)))
        return k

    def _sturges_rule(self, data):
        """
        Calculate number of histogram bins using the Sturges' Rule.
        """
        x = np.asarray(data)
        n = x.size
        k = int(np.ceil(np.log2(n)) + 1)
        return k

    def _scotts_rule(self, data):
        """
        Calculate number of histogram bins using the Square-root Rule.
        """
        x = np.asarray(data)
        n = x.size
        h = 3.49 * np.std(x) * n**(-1 / 3)
        k = int(np.ceil((x.max() - x.min()) / h))
        return k

    def _freedman_diaconis_rule(self, data):
        """
        Calculate number of histogram bins using Freedman-Diaconis rule.
        """
        x = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("data must be one-dimensional")
        n = x.size
        if n < 2:
            k = 1
        else:
            q75, q25 = np.percentile(x, [75, 25])
            iqr = q75 - q25
            h = 2 * iqr * n**(-1 / 3)
            if h == 0:
                k = self._square_root_rule(x)
            else:
                k = int(np.ceil((x.max() - x.min()) / h))
        return k

    def _freedman_diaconis_rule(self, data):
        """
        Calculate number of hist bins using Freedman-Diaconis rule.

        https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
        https://stats.stackexchange.com/questions/798/
        https://stackoverflow.com/questions/23228244/how-do-you-find-the-iqr-in-numpy

        The Freedman-Diaconis rule can be used to select the width of the bins
        to be used in a histogram.

        The general equation for the rule is
                Bin width = 2 * IQR(x) * n^(-1/3),
        where IQR(x) is the interquartile range of the data and n is the number
        of observations in the sample x.

        The number of bins is then
                Number of bins = (max - min) / Bin width,
        where max is the maximum and min is the minimum value of the data.
        """

        '''
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("data should be one-dimensional")

        n = data.size
        if n < 4:
            raise ValueError("data should have more than three entries")

        v25, v75 = np.percentile(data, [25, 75])
        dx = 2 * (v75 - v25) / (n ** (1 / 3))
        '''
        x = np.asarray(data)
        n = len(x)
        if n < 2:
            n_bins = 1
        else:
            q75, q25 = np.percentile(x, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr * n**(-1 / 3)
            n_bins = int(np.sqrt(x.size)) if bin_width == 0 else int(
                np.ceil((x.max() - x.min()) / bin_width))
        return n_bins

    def _set_plot_style(self):
        params = {'legend.fontsize': 'large',
                  'axes.labelsize': 'large',
                  'axes.titlesize': 'large',
                  'xtick.labelsize': 'large',
                  'ytick.labelsize': 'large',
                  'legend.fontsize': 'large',
                  'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.rc('text', usetex=True)

    def _add_histplot(self, data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values):
        n_bins = self._freedman_diaconis_rule(data)
        ax.hist(data, density=density, histtype='bar', edgecolor=None,
                color='steelblue', alpha=0.5, bins=n_bins, label="Accepted samples")
        ax.axvline(
            point_estimates[index], color='b', label="Point estimate")
        if true_vals_bool:
            ax.axvline(
                true_parameter_values[index], color='r', linestyle='--', label="Groundtruth")
        ax.set_xlabel(self.labels[index])
        ax.set_title("Histogram of accepted " + self.labels[index])

    def histplot(self, density=True, show=True, dpi=120, path_to_save=None, true_parameter_values=None):
        """
        histogram(s) of sampled parameter(s)
        """
        # run checks
        self._check_journal_status()
        true_vals_bool = False
        if true_parameter_values is not None:
            self._check_true_parameter_values(true_parameter_values)
            true_vals_bool = True

        # get sampled parameters
        *data, = self._get_params_as_arrays()
        point_estimates = self._point_estimates()

        fig = plt.figure(figsize=(8, 6), tight_layout=True, dpi=dpi)
        self._set_plot_style()

        N = self._n_parameters

        if N == 1:
            ax = plt.subplot(111)
            legend_position = 0
            index = 0
            self._add_histplot(
                data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values)
        else:
            if N == 2 or N == 4:
                cols = 2
                legend_position = 1
            else:
                cols = 3
                legend_position = 2
            rows = int(np.ceil(N / cols))
            gs = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)
            for index, data in enumerate(data):
                ax = fig.add_subplot(gs[index])
                self._add_histplot(
                    data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values)

        handles, labels = plt.gca().get_legend_handles_labels()
        if true_vals_bool:
            order = [2, 0, 1]
        else:
            order = [1, 0]

        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='center left',
                   bbox_to_anchor=(1.04, 0.5),
                   fancybox=True,
                   borderaxespad=0.1,
                   ncol=1
                   )

        if path_to_save is not None:
            fig.savefig(path_to_save, dpi=dpi)
        if show:
            plt.show()

    def adjusted_histplot():
        # regression adjusted
        pass

    def kdeplot(self, kernel="gaussian"):
        ax[1].plot(x, kernel.evaluate(x), label="approximate posterior")
        pass

    def distplot(self, kde=True, kde_kwds=None, ax=None):
        """
        """
        if ax is None:
            ax = plt.gca()
        pass

    def posterior_kde(self, kernel="gaussian"):
        pass

    @ property
    def summary(self):
        pass

    @ property
    def print_summary(self):
        pass

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

    def load(self, filename):
        with open(filename, 'rb') as input:
            journal = pickle.load(input)
        return journal

    '''
    @staticmethod
    def run_lra(
        theta: torch.Tensor,
        x: torch.Tensor,
        observation: torch.Tensor,
        sample_weight=None,
    ) -> torch.Tensor:
        """Return parameters adjusted with linear regression adjustment.
        Implementation as in Beaumont et al. 2002: https://arxiv.org/abs/1707.01254
        """

        theta_adjusted = theta
        for parameter_idx in range(theta.shape[1]):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(
                X=x,
                y=theta[:, parameter_idx],
                sample_weight=sample_weight,
            )
            theta_adjusted[:, parameter_idx] += regression_model.predict(
                observation.reshape(1, -1)
            )
            theta_adjusted[:, parameter_idx] -= regression_model.predict(x)

        return theta_adjusted
    '''
