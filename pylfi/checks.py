#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def check_1D_data(data):
    """Check that data is one-dimensional"""
    if data.ndim != 1:
        raise ValueError("data must be one-dimensional")
