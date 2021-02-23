#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as stats


class ToyModels:

    def __init__(self):
        pass

    @staticmethod
    def gaussian_unkwown_variance(var, n_samples):
        return stats.norm(loc=0., scale=np.sqrt(var)).rvs(size=n_samples)

    @staticmethod
    def gaussian_unkwown_mean(mean, n_samples):
        pass

    @staticmethod
    def binomial_coin_toss(n_samples):
        pass

    @staticmethod
    def poisson_process(n_samples):
        pass

    @staticmethod
    def gaussian_model(mu, sigma, n_samples):
        return stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)

    @staticmethod
    def multinormal(n_samples):
        pass


class SimpleODE:

    def __init__(self):
        pass

    @staticmethod
    def exponential_decay(n_samples):
        pass

    @staticmethod
    def lotka_volterra(n_samples):
        pass


class NeuroModels:

    def __init__(self):
        pass

    @staticmethod
    def hodgkin_huxley(n_samples):
        pass
