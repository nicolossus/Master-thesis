#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import matplotlib.pyplot as plt
import numpy as np
from abcpy.backends import BackendDummy, BackendMPI
from abcpy.continuousmodels import Normal, Uniform
from abcpy.distances import Euclidean, LogReg, Wasserstein
from abcpy.inferences import PMCABC, SMCABC, RejectionABC
from abcpy.perturbationkernel import DefaultKernel
from abcpy.statistics import Identity
from abcpy.statisticslearning import Semiautomatic, StatisticsLearningNN
from model import HHSimulator
from pylfi.models import constant_stimulus

# ignore overflow warnings; occurs with certain np.exp() evaluations
np.warnings.filterwarnings('ignore', 'overflow')


def setup_backend(parallel=False):
    if parallel:
        backend = BackendMPI()
    else:
        backend = BackendDummy()
    return backend


def infer_parameters(backend, scheme='rejection', n_samples=250, n_samples_per_param=10, logging_level=logging.WARN):
    """Perform inference for this example.
    Parameters
    ----------
    backend
        The parallelization backend
    steps : integer, optional
        Number of iterations in the sequential PMCABC algoritm ("generations"). The default value is 3
    n_samples : integer, optional
        Number of posterior samples to generate. The default value is 250.
    n_samples_per_param : integer, optional
        Number of data points in each simulated data set. The default value is 10.
    Returns
    -------
    abcpy.output.Journal
        A journal containing simulation results, metadata and optionally intermediate results.
    """
    logging.basicConfig(level=logging_level)

    # experimental setup
    T = 50.           # simulation time
    dt = 0.025        # time step
    I_amp = 0.32      # stimulus amplitude
    r_soma = 40       # radius of soma
    threshold = -55   # AP threshold

    # input stimulus
    stimulus_dict = constant_stimulus(
        I_amp=I_amp, T=T, dt=dt, t_stim_on=10, t_stim_off=40, r_soma=r_soma)
    I = stimulus_dict["I"]
    #I_stim = stimulus_dict["I_stim"]

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
    obs_data = hh_simulator.forward_simulate([gbar_K_true, gbar_Na_true])

    # define statistics
    statistics_calculator = Identity()

    # Learn the optimal summary statistics using Semiautomatic summary selection
    statistics_learning = Semiautomatic(
        [hh_simulator], statistics_calculator, backend,
        n_samples=1000, n_samples_per_param=1, seed=42)
    new_statistics_calculator = statistics_learning.get_statistics()

    # define distance
    distance_calculator = Euclidean(new_statistics_calculator)

    # define kernel
    kernel = DefaultKernel([gbar_K, gbar_Na])

    # define sampling scheme
    if scheme == 'rejection':
        sampler = RejectionABC(
            [hh_simulator], [distance_calculator], backend, seed=42)
        # sample from scheme
        epsilon = 2.
        journal = sampler.sample([obs_data], n_samples,
                                 n_samples_per_param, epsilon)

    elif scheme == 'smc':
        sampler = SMCABC([hh_simulator], [distance_calculator],
                         backend, kernel, seed=42)
        # sample from scheme
        steps = 3
        journal = sampler.sample([obs_data], steps, n_samples,
                                 n_samples_per_param)
    elif scheme == 'pmc':
        sampler = PMCABC([hh_simulator], [distance_calculator],
                         backend, kernel, seed=42)
        # sample from scheme
        steps = 3
        eps_arr = np.array([2.])
        epsilon_percentile = 10
        journal = sampler.sample([obs_data], steps, eps_arr, n_samples,
                                 n_samples_per_param, epsilon_percentile)

    return journal


def analyse_journal(journal, fn="posterior.png"):
    # output parameters and weights

    # print(journal.get_parameters())
    # print(journal.get_weights())

    # do post analysis
    print(journal.posterior_mean())
    print(journal.posterior_cov())

    # print configuration
    print(journal.configuration)

    # plot posterior
    journal.plot_posterior_distr(path_to_save=fn)

    # save and load journal
    journal.save("experiments.jnl")

    #from abcpy.output import Journal
    #new_journal = Journal.fromFile('experiments.jnl')


def setUpModule():
    '''
    If an exception is raised in a setUpModule then none of
    the tests in the module will be run.

    This is useful because the slaves run in a while loop on initialization
    only responding to the master's commands and will never execute anything else.
    On termination of master, the slaves call quit() that raises a SystemExit().
    Because of the behaviour of setUpModule, it will not run any unit tests
    for the slave and we now only need to write unit-tests from the master's
    point of view.
    '''
    setup_backend()


if __name__ == "__main__":
    backend = setup_backend(parallel=True)
    journal = infer_parameters(
        backend, scheme='smc', logging_level=logging.INFO)
    analyse_journal(journal, "posterior_smc.png")
