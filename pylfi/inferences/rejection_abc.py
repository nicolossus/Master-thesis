#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
from pylfi.inferences import ABCBase
from pylfi.journal import Journal


class RejectionABC(ABCBase):

    def __init__(self, observation, simulator, priors, distance='l2'):
        """
        simulator : callable
            simulator model
        summary_calculator : callable, defualt None
            summary statistics calculator. If None, simulator should output
            sum stat
        distance : str
            Can be a custom function or one of l1, l2, mse
        distance_metric : callable
            discrepancy measure
        """

        # self._obs = observation
        # self._simulator = simulator  # model simulator function
        # self._priors = priors
        # self._distance = distance    # distance metric function
        super().__init__(
            observation=observation,
            simulator=simulator,
            priors=priors,
            distance=distance,
        )

    def __call__(self, num_simulations, epsilon, lra=False):
        journal = self.sample(num_simulations, epsilon, lra)
        return journal

    def sample(self, num_simulations, epsilon, lra=False):
        """
        add **kwargs for simulator call

        Pritchard et al. (1999) algorithm

        n_samples: integer
            Number of samples to generate

        lra bool, Whether to run linear regression adjustment as in Beaumont et al. 2002
        """

        _inference_scheme = "Rejection ABC"
        self.logger.info(f"Initializing {_inference_scheme} inference scheme.")
        n_sims = num_simulations

        journal = Journal()  # journal instance
        journal._start_journal()

        journal._add_config(self._simulator, _inference_scheme,
                            self._distance, num_simulations, epsilon)
        journal._add_parameter_names(self._priors)

        # draw thetas from priors
        thetas = np.array([prior.rvs(size=(n_sims,))
                          for prior in self._priors])
        # simulated
        sims = np.array([self._simulator(*thetas)
                        for thetas in np.stack(thetas, axis=-1)])
        # distances
        distances = np.array([self._distance(self._obs, sim) for sim in sims])
        # acceptance criterion
        is_accepted = distances < epsilon

        # accepted simulations
        num_accepted = is_accepted.sum().item()
        thetas_accepted = thetas[:, is_accepted]
        dist_accepted = distances[is_accepted]
        sims_accepted = sims[is_accepted]

        for i, thetas in enumerate(np.stack(thetas_accepted, axis=-1)):
            journal._add_accepted_parameters(thetas)
            journal._add_distance(dist_accepted[i])
            journal._add_sumstat(sims_accepted[i])

        journal._add_sampler_summary(n_sims, num_accepted)

        self.logger.info(f"Accepted {num_accepted} of {n_sims} simulations.")

        if lra:
            # self.logger.info("Running Linear regression adjustment.")
            # journal.do
            pass

        return journal
