#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class DistanceMetrics:

    def __init__(self):
        pass

    @staticmethod
    def euclidean(sim_data, obs_data):
        return np.sqrt(np.sum((sim_data - obs_data) * (sim_data - obs_data)))
