#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

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


def beta(a, b, size=None):
    return np.random.default_rng().beta(a, b, size)


def binomial(n, p, size=None):
    return np.random.default_rng().binomial(n, p, size)


def chisquare(df, size=None):
    return np.random.default_rng().chisquare(df, size)


def dirichlet(alpha, size=None):
    return np.random.default_rng().dirichlet(alpha, size)


def exponential(scale=1.0, size=None):
    return np.random.default_rng().exponential(scale, size)


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

    def pdf(self, x):
        pass

    def rvs(self, size=None):
        return self._rng.uniform(self._low, self._high, size)

    @property
    def name(self):
        return self._name

    @property
    def extradoc(self):
        pass
