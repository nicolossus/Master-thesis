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
