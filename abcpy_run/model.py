#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
from abcpy.continuousmodels import (Continuous, InputConnector,
                                    ProbabilisticModel)
from pylfi.models import HodgkinHuxley


class HHSimulator(ProbabilisticModel, Continuous):
    """
    Hodgkin-Huxley model simulator conforming to ABCpy's API.

    In this inference task, we want to infer the conductance parameters
    `gbar_K` and `gbar_Na`.
    """

    def __init__(self, parameters, stimulus, T, dt, name='HHSimulator'):
        """
        Initialize derived class to properly connect it to its input models.

        It accepts as input an InputConnector object that fully specifies how
        to connect all parent models to the current model.

        Parameters
        ----------
        parameters : list
            A list of input parameters to connect.
        stimulus : array of shape (int(T/dt)+1) or callable with signature '(t)'
            The input stimulus
        T : float
            Simulation time
        dt : float
            Time step
        name : str
            A human readable name for the model.
        """
        self.I = stimulus
        self.T = T
        self.dt = dt
        self.hh = HodgkinHuxley()
        input_parameters = InputConnector.from_list(parameters)
        super(HHSimulator, self).__init__(input_parameters, name)

    def _check_input(self, input_values):
        """
        Check whether the input parameters are compatible with the underlying model.

        Parameters
        ----------
        input_values : list
            A list of numbers that are the concatenation of all parent model
            outputs in the order specified by the InputConnector object that
            was passed during initialization.

        Returns
        -------
        boolean
            True if the fixed value of the parameters can be used as input
            for the current model. False otherwise.
        """
        if len(input_values) != 2:
            return False
        return True

    def forward_simulate(self, input_values, k=1, rng=np.random.RandomState()):
        """
        Provides the output from a forward simulation of the current model.

        Parameters
        ----------
        input_values : list
            A list of numbers that are the concatenation of all parent model
            outputs in the order specified by the InputConnector object that
            was passed during initialization.
        k : int
            The number of forward simulations that should be run.
        rng : Random number generator
            Defines the random number generator to be used. The default value
            uses a random seed to initialize the generator.

        Returns
        -------
        results : list
            A list of k elements, where each element is of type numpy.array and
            represents the result of a single forward simulation.
        """
        gbar_K = input_values[0]
        gbar_Na = input_values[1]

        # set proposed conductance parameters
        self.hh.gbar_K = gbar_K
        self.hh.gbar_Na = gbar_Na

        self.hh.solve(self.I, self.T, self.dt)

        # duplicate for the required k
        X = np.stack([self.hh.V] * k)
        results = [x for x in X]

        return results

    def _check_output(self, values):
        """
        Checks whether values contains a reasonable output of the current model.

        In the case of inference on mechanistic models, it is safe to
        omit the check and return True.
        """
        return True

    def get_output_dimension(self):
        """
        Provides the output dimension of the current model.
        """
        return int(self.T / self.dt) + 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gbar_K = 36
    gbar_Na = 120
    T = 50.
    dt = 0.025
    k = 2
    rng = np.random.RandomState(seed=42)

    def stimulus(t):
        if 10 < t < 40:
            return 10
        return 0

    hh_simulator = HHSimulator([gbar_K, gbar_Na], stimulus, T, dt)
    out = hh_simulator.forward_simulate([gbar_K, gbar_Na], k, rng)
    plt.plot(out[0])
    plt.show()
