#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
from abcpy.continuousmodels import (Continuous, InputConnector,
                                    ProbabilisticModel)
from abcpy.statistics import Statistics
from pylfi.features import SpikingFeatures
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


class Features(Statistics):
    """
    This class wraps the `SpikingFeatures` class such that it conforms to
    ABCpy's API.
    """
    # def __init__(self, V, t, stim_duration, t_stim_on, threshold=-55):

    def __init__(self, t, stim_duration, t_stim_on, threshold=-55):
        """
        Parameters
        ----------
        degree : integer, optional
            Of polynomial expansion. The default value is 2 meaning second order polynomial expansion.
        cross : boolean, optional
            Defines whether to include the cross-product terms. The default value is True, meaning the cross product term
            is included.
        previous_statistics : Statistics class, optional
            It allows pipelining of Statistics. Specifically, if the final statistic to be used is determined by the
            composition of two Statistics, you can pass the first here; then, whenever the final statistic is needed, it
            is sufficient to call the `statistics` method of the second one, and that will automatically apply both
            transformations.
        """
        self.t = t
        self.duration = stim_duration
        self.t_stim_on = t_stim_on
        self.threshold = threshold

    def statistics(self, data):
        """
        Parameters
        ----------
        data: python list
            Contains n data sets with length p.
        Returns
        -------
        numpy.ndarray
            nx(p+degree*p+cross*nchoosek(p,2)) matrix where for each of the n data points with length p,
            (p+degree*p+cross*nchoosek(p,2)) statistics are calculated.
        """

        features = SpikingFeatures(
            data[0], self.t, self.duration, self.t_stim_on, self.threshold)

        print(f"{features.n_spikes=}")
        result = [features.spike_rate]
        # result = [features.average_AP_overshoot]
        #result = [features.average_AHP_depth]

        result = self._check_and_transform_input(result)

        return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from abcpy.backends import BackendDummy, BackendMPI
    from abcpy.continuousmodels import Normal, Uniform
    from abcpy.distances import Euclidean, LogReg, Wasserstein
    from abcpy.inferences import PMCABC, SMCABC, RejectionABC
    from abcpy.perturbationkernel import DefaultKernel
    from abcpy.statistics import Identity
    from abcpy.statisticslearning import Semiautomatic, StatisticsLearningNN
    # from model import Features, HHSimulator
    from pylfi.models import constant_stimulus
    '''
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
    '''

    backend = BackendDummy()

    # experimental setup
    T = 120.           # simulation time
    dt = 0.025        # time step
    I_amp = 0.32      # stimulus amplitude
    r_soma = 40       # radius of soma
    threshold = -55   # AP threshold

    # input stimulus
    stimulus_dict = constant_stimulus(
        I_amp=I_amp, T=T, dt=dt, t_stim_on=10, t_stim_off=100, r_soma=r_soma)
    I = stimulus_dict["I"]
    # I_stim = stimulus_dict["I_stim"]

    # true parameters
    gbar_K_true = 36
    gbar_Na_true = 120

    gbar_K_std = 5
    gbar_Na_std = 5

    # define priors
    gbar_K = Normal([[gbar_K_true], [gbar_K_std]], name='gbar_K')
    gbar_Na = Normal([[gbar_Na_true], [gbar_Na_std]], name='gbar_Na')

    # define the model
    hh_simulator = HHSimulator([gbar_K, gbar_Na], I, T, dt)

    # observed data
    rseed = 42
    rng = np.random.RandomState(rseed)
    n_samples = int(T / dt) + 1
    noise = rng.normal(loc=0, scale=0.05, size=(n_samples))
    obs_data = hh_simulator.forward_simulate([gbar_K_true, gbar_Na_true])

    # define statistics
    # statistics_calculator = Identity()
    t = stimulus_dict["t"]
    stim_duration = stimulus_dict["duration"]
    t_stim_on = stimulus_dict["t_stim_on"]
    statistics_calculator = Features(t, stim_duration, t_stim_on)

    # define distance
    distance_calculator = Euclidean(statistics_calculator)
    distance_calculator2 = Wasserstein(statistics_calculator)
    distance_calculator3 = LogReg(statistics_calculator)

    # checking
    sim_data = hh_simulator.forward_simulate(
        [gbar_K_true + 3, gbar_Na_true + 4])

    #obs_data = [obs_data[0] / np.max(obs_data[0]) + noise]
    #sim_data = [sim_data[0] / np.max(sim_data[0])]

    s1 = statistics_calculator.statistics(obs_data)
    s2 = statistics_calculator.statistics(sim_data)
    dist = distance_calculator.distance(obs_data, sim_data)
    dist2 = distance_calculator2.distance(obs_data, sim_data)
    dist3 = distance_calculator3.distance(obs_data, sim_data)
    print(s1)
    print(s2)
    print(dist)
    print(dist2)
    print(dist3)
    # plt.plot(t, obs_data[0] / np.max(obs_data[0]), label='obs')
    # plt.plot(t, sim_data[0] / np.max(sim_data[0]), label='sim')
    plt.plot(t, obs_data[0], label='obs')
    plt.plot(t, sim_data[0], label='sim')
    plt.legend()
    plt.show()

    '''
    # inference scheme
    sampler = RejectionABC(
        [hh_simulator], [distance_calculator], backend, seed=42)
    # sample from scheme
    n_samples = 250
    n_samples_per_param = 10
    epsilon = 2.
    journal = sampler.sample([obs_data], n_samples,
                             n_samples_per_param, epsilon)
    print(journal.posterior_mean())
    print(journal.posterior_cov())

    # print configuration
    print(journal.configuration)

    # plot posterior
    journal.plot_posterior_distr(path_to_save='posterior.png')
    '''
