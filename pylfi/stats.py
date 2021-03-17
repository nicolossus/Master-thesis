#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def iqr(data, norm=False):
    """Calculate inter-quartile range (IQR) of the given data. Returns
    normalized IQR(x) if keyword norm=True. 
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if norm:
        normalize = 1.349  # normalize = norm.ppf(.75) - norm.ppf(.25)
        iqr /= normalize
    return iqr


def covmatrix():
    pass


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    data = rng.randn(100)
    print(iqr(data))
    print(iqr(data, norm=True))
