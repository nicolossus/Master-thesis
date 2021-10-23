import pickle

import numpy as np


def save_spiketrain(spiketrain, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(spiketrain, fp)


def load_spiketrain(filename):
    with open(filename, 'rb') as fp:
        spiketrain = pickle.load(fp)
    return spiketrain


def slice_spiketrains(spiketrains, t_start=None, t_stop=None):

    spiketrains_slice = []
    for spiketrain in spiketrains:
        if t_start is None:
            t_start = spiketrain.t_start
        if t_stop is None:
            t_stop = spiketrain.t_stop

        spiketrain_slice = spiketrain[np.where(
            (spiketrain > t_start) & (spiketrain < t_stop))]
        spiketrain_slice.t_start = t_start
        spiketrain_slice.t_stop = t_stop
        spiketrains_slice.append(spiketrain_slice)
    return spiketrains_slice
