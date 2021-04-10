#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def observed_data(seed=None):

    if seed is not None:
        self.rng = np.random.RandomState(seed=seed)
    else:
        self.rng = np.random.RandomState()

    pass


def plot_observed_data():
    pass


def HH_simulator():
    pass


def plot_simulation():
    pass


def HH_simulator(gbar_K, gbar_Na, statistic='ST'):
    """HH wrapper that returns set of summary statistics

    statistic:
    'ST' - Spike Time

    'All' - set of all sum stats

    todo
    ----
    move to top of file

    """
    hh = HodgkinHuxley()
    hh.gbar_K = gbar_K
    hh.gbar_Na = gbar_Na
