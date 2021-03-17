#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np
from checks import check_1D_data, check_bandwidth, check_kernel
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity

# warnings.filterwarnings("ignore")

'''
perhaps scrap sklearn and use this instead:
https://github.com/tommyod/KDEpy
'''


class KDE:
    """
    bandwidth 'auto': grid with 10 uniformly spaced bandwidth values between
    1e-3 and 1e0
    """

    def __init__(self, data, bandwidth=1.0, kernel='gaussian', **kwargs):

        check_bandwidth(bandwidth)
        check_kernel(kernel)

        self._data = data
        self._bw = bandwidth
        self._kernel = kernel

        self._data_len = len(self._data)

        self._grid_scores = None
        self._best_params = None
        self._optimal_bw = None
        self._optimal_kernel = None

        self._select_kde(**kwargs)
        self._fit()

    def __call__(self, x):
        return self.density(x)

    def _select_kde(self, **kwargs):
        """
        Initialize KDE
        """

        if isinstance(self._bw, str):
            if self._bw == "auto":
                self._bw = np.logspace(-4, 1, 100)
            elif self._bw == "scott":
                self._bw = [self._scotts_bw_rule()]
            elif self._bw == "silverman":
                self._bw = [self._silverman_bw_rule()]
        elif isinstance(self._bw, (int, float)):
            self._bw = [self._bw]

        if isinstance(self._kernel, str):
            if self._kernel == 'auto':
                self._kernel = ['gaussian', 'tophat', 'epanechnikov',
                                'exponential', 'linear', 'cosine']
            else:
                self._kernel = [self._kernel]

        self._kde = self._optimal_kde(**kwargs)

    def _scotts_bw_rule(self):
        """See https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/bandwidths.py"""
        pass

    def _silverman_bw_rule(self):
        """see same as scott's"""
        pass

    def _optimal_kde(self, **kwargs):
        """
        Hyper-parameter estimation for optimal bandwidth and kernel using
        grid search with cross-validation
        """

        grid = GridSearchCV(KernelDensity(**kwargs),
                            {'bandwidth': self._bw,
                             'kernel': self._kernel},
                            # scoring='mutual_info_score',
                            # scoring=_scoring,
                            # cv=LeaveOneOut(),
                            # cv=5
                            )
        return grid

    def _fit(self):
        # print(self._kde)
        self._kde.fit(self._data[:, None])
        self._grid_scores = self._kde.cv_results_
        self._best_params = self._kde.best_params_
        self._optimal_bw = self._kde.best_estimator_.bandwidth
        self._optimal_kernel = self._kde.best_estimator_.kernel
        self._kde = self._kde.best_estimator_

    def density(self, x):
        x = np.asarray(x)
        check_1D_data(x)
        log_dens = self._kde.score_samples(x[:, None])
        return np.exp(log_dens)

    def sample(self, n_samples):
        return self._kde.sample(n_samples)

    def sample2(self, n_samples, rseed=42):
        rng = np.random.RandomState(rseed)
        i = rng.uniform(0, self._data_len, size=n_samples)

    @property
    def grid_scores(self):
        return self._grid_scores

    @property
    def best_params(self):
        '''
        # redundant with current implementation
        if self._best_params is None:
            msg = ("Best parameters only available after fitting")
            raise ValueError(msg)
        '''
        return self._best_params

    @property
    def bandwidth(self):
        return self._optimal_bw

    @property
    def kernel(self):
        return self._optimal_kernel

    def plot_kernels(self):
        # plot available kernels
        # see this instead:
        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
        kernels = ['cosine', 'epanechnikov',
                   'exponential', 'gaussian', 'linear', 'tophat']
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
        plt_ind = np.arange(6) + 231

        for k, ind in zip(kernels, plt_ind):
            kde_model = KernelDensity(kernel=k)
            kde_model.fit([[0]])
            score = kde_model.score_samples(np.arange(-2, 2, 0.1)[:, None])
            plt.subplot(ind)
            plt.fill(np.arange(-2, 2, 0.1)[:, None], np.exp(score), c='blue')
            plt.title(k)

    def plot_grid_search(self):
        scores_mean = self._grid_scores['mean_test_score']
        scores_mean = np.array(scores_mean).reshape(
            len(self._kernel), len(self._bw))

        scores_std = self._grid_scores['std_test_score']
        scores_std = np.array(scores_std).reshape(
            len(self._kernel), len(self._bw))

        fig, ax = plt.subplots(1, 1)

        for idx, kernel_name in enumerate(self._kernel):
            ax.plot(np.log10(self._bw),
                    scores_mean[idx, :], '-', label=kernel_name)
        ax.set_title("Grid Search Scores")
        ax.set_xlabel("Bandwidth")
        ax.set_ylabel('CV Average Score')
        ax.legend(loc="best")
        ax.grid('on')


# Maybe include


def _scoring(estimator, X):
    """
    The cosine, linear and tophat kernels might give a runtime warning due
    to some scores resulting in -inf values. This issue is addressed by
    writing a custom scoring function for GridSearchCV()
    """
    scores = estimator.score_samples(X)
    # Remove -inf
    scores = scores[scores != float('-inf')]
    #scores = scores[np.isfinite(scores)]
    # Return the mean values
    return np.abs(np.mean(scores))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from sklearn.decomposition import PCA

    def make_data(N, f=0.3, rseed=1):
        rand = np.random.RandomState(rseed)
        x = rand.randn(N)
        #x[int(f * N):] += 5
        return x

    def make_data2(N):
        groundtruth = 2.0
        likelihood = stats.norm(loc=0., scale=np.sqrt(groundtruth))
        obs_data = likelihood.rvs(size=N)
        return obs_data

    data = make_data2(1000)

    # observed data

    x = np.linspace(-4, 10, 1000)
    #kde = KDE(data, bandwidth='auto', kernel='auto')
    kde = KDE(data, bandwidth='auto', kernel=[
              'gaussian', 'epanechnikov'])

    best_params = kde.best_params
    print('Optimal params:', best_params)

    grid_scores = kde.grid_scores
    kde.plot_grid_search()
    #plt.ylim(-0.5, 0.5)
    plt.show()
    # print(grid_scores)
    #plot.grid_search(grid_scores, change='n_estimators', kind='bar')

    density = kde.density(x)
    density2 = kde(x)
    kde_samples = kde.sample(1000)

    sample_mean = np.mean(data)
    kde_mean = np.mean(kde_samples)
    print('sample mean:', sample_mean)
    print('kde mean:', kde_mean)
    plt.figure()
    plt.hist(data, density=True)
    plt.plot(x, density, color='r')
    plt.axvline(kde_mean, color='k', ls=':')
    plt.axvline(sample_mean, color='k', ls='--')
    plt.show()
