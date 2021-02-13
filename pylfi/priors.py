#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import scipy.stats as stats
from numpy.random import default_rng

# https://numpy.org/doc/stable/reference/random/index.html#random-quick-start
"""
# Do this (new version)
from numpy.random import default_rng
rng = default_rng()
vals = rng.standard_normal(10)
more_vals = rng.standard_normal(10)

# instead of this (legacy version)
from numpy import random
vals = random.standard_normal(10)
more_vals = random.standard_normal(10)
"""

# TODO: make this module more streamlined and coherent
# consider changing from numpy.random to scipy.stats despite slower performance
# add method for plotting the prior


class Uniform:
    def __init__(self, low=0.0, high=1.0, name="Uniform"):
        """
        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        self._rng = default_rng()
        self._low = low
        self._high = high
        self._name = name

    def rvs(self, size=None):
        return self._rng.uniform(self._low, self._high, size)

    def pdf(self, x):
        pass

    @property
    def name(self):
        return self._name

    @property
    def extradoc(self):
        pass


class Binomial:
    def __init__(self, n, p, name="Binomial"):
        """
        n is the number of trials, p is the probability of success

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        self._rng = default_rng()
        self._n = n
        self._p = p
        self._name = name

    def rvs(self, size=None):
        return self._rng.binomial(self._n, self._p, size)

    def pdf(self, x):
        pass

    @property
    def name(self):
        return self._name

    @property
    def extradoc(self):
        pass


class NegativeBinomial:
    def __init__(self, n, p, name="Negative Binomial"):
        """
        n is the number of trials, p is the probability of success

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        self._rng = default_rng()
        self._n = n
        self._p = p
        self._name = name

    def rvs(self, size=None):
        return self._rng.negative_binomial(self._n, self._p, size)

    def pdf(self, x):
        pass

    @property
    def name(self):
        return self._name

    @property
    def extradoc(self):
        pass


class Beta:
    def __init__(self, alpha, beta, name="Beta"):
        """
        a float or array_like of floats
            Alpha, positive (>0).

        b float or array_like of floats
            Beta, positive (>0).

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        self._rng = default_rng()
        self._a = alpha
        self._b = beta
        self._name = name

    def rvs(self, size=None):
        return self._rng.beta(self._a, self._b, size)

    def pdf(self, x):
        pass

    @property
    def name(self):
        return self._name

    @property
    def extradoc(self):
        pass


class Exponential:
    def __init__(self, beta=1.0, name="Exponential"):
        """
        The scale parameter, \beta = 1/\lambda. Must be non-negative.

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        self._rng = default_rng()
        self._scale = beta
        self._name = name

    def rvs(self, size=None):
        return self._rng.exponential(self._scale, size)

    def pdf(self, x):
        pass

    @property
    def name(self):
        return self._name

    @property
    def extradoc(self):
        pass


class Gamma:
    def __init__(self, k, theta=1.0, name="Gamma"):
        """
        k is the shape and \theta the scale

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        self._rng = default_rng()
        self._shape = k
        self._scale = theta
        self._name = name

    def rvs(self, size=None):
        return self._rng.gamma(self._shape, self._scale, size)

    def pdf(self, x):
        pass

    @property
    def name(self):
        return self._name

    @property
    def extradoc(self):
        pass


class InvGamma:
    def __init__(self, k, loc=0.0, beta=1.0, name="Inverse Gamma"):
        """
        k is the shape and \theta the scale

        name: string
            The name that should be given to the probabilistic model in the journal file.
        """

        self._rng = default_rng()
        self._shape = k
        self._loc = loc
        self._scale = beta
        self._name = name

    def rvs(self, size=1):
        return stats.invgamma.rvs(self._shape, self._loc, self._scale, size)

    def pdf(self, x):
        pass

    @property
    def name(self):
        return self._name

    @property
    def extradoc(self):
        pass


def chisquare(df, size=None):
    return np.random.default_rng().chisquare(df, size)


def dirichlet(alpha, size=None):
    return np.random.default_rng().dirichlet(alpha, size)


# multinormal


if __name__ == "__main__":
    theta_uniform = Uniform(low=2, high=2.5)
    print(f"{theta_uniform.rvs(5)=}")

    theta_binomial = Binomial(n=10, p=0.5)
    print(f"{theta_binomial.rvs(5)=}")

    theta_neg_binomial = NegativeBinomial(n=10, p=0.5)
    print(f"{theta_neg_binomial.rvs(5)=}")

    theta_beta = Beta(alpha=2, beta=3)
    print(f"{theta_beta.rvs(5)=}")

    theta_exponential = Exponential(beta=1)
    print(f"{theta_exponential.rvs(5)=}")

    theta_gamma = Gamma(k=1, theta=1)
    print(f"{theta_gamma.rvs(5)=}")

    theta_invgamma = InvGamma(k=1, loc=0, beta=1.0)
    print(f"{theta_invgamma.rvs(5)=}")
