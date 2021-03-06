#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, Union

from pylfi.distances import euclidean
from pylfi.utils.checks import check_distance_str


class ABCBase(metaclass=ABCMeta):
    def __init__(
        self,
        observation,
        simulator: Callable,
        priors,
        distance: Union[str, Callable] = "l2"
    ) -> None:
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
        self._obs = observation
        self._simulator = simulator
        self._priors = priors

        # Select distance function.
        if callable(distance):
            self._distance = distance
        elif isinstance(distance, str):
            check_distance_str(distance)
            self._distance = self._choose_distance(distance)
        else:
            raise TypeError()

        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def sample(self):
        """To be overwritten by sub-class: should implement sampling from
        inference scheme and return journal.

        Returns
        -------
        pylfi.journal
            Journal
        """

        raise NotImplementedError

    @staticmethod
    def _choose_distance(distance):
        """Return distance function for given distance type."""
        if distance == 'l1':
            return None
        elif distance == 'l2':
            return euclidean
        elif distance == 'mse':
            return None

    @staticmethod
    def run_lra():
        """Linear regression adjustment as in Beaumont et al. 2002.
        """
        pass
