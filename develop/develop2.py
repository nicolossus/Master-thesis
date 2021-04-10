
import math

import numpy as np
from numpy.random import default_rng
from scipy.stats import invgamma, norm

alpha = 60
beta = 130
prior1 = invgamma(alpha, loc=0, scale=beta)
prior2 = invgamma(alpha, loc=0, scale=beta)
prior3 = invgamma(alpha, loc=0, scale=beta)

priors = [prior1, prior2, prior3]

thetas = [prior.rvs(size=1) for prior in priors]
# print(thetas)
thetas = [prior.rvs(size=5) for prior in priors]
# print(thetas)
#thetas = prior.rvs(size=Nsims)

low = 0
high = 1
size = 10

rng = default_rng()
uniform = rng.uniform(low, high, size)
# print(type(uniform))
# print(uniform)

params = (np.array([1, 2, 3]), np.array([1, 2, 3]))
print(params)

params += (np.array([1, 2, 3]))

print(params)

params = (np.asarray(samples, float) for samples in thetas)

params = (np.asarray(samples, float).squeeze() if np.asarray(samples, float).ndim >
          1 else np.asarray(samples, float).squeeze() for samples in thetas)

*samples, = params

# print(samples)
# print(len(samples))

N = [x for x in range(1, 10)]

for n in N:
    if n == 1:
        cols = 1
    elif n == 2:
        cols = 2
    elif n == 4:
        cols = 2
    else:
        cols = 3
    rows = int(np.ceil(n / cols))
    print(n, cols, rows)

a = None
b = []
for i in range(2):
    b.append(a)
print(b)


def rvs(self, size=1):
    return self.uniform.choice

    '''
    def rejection_sampler(obs_data, prior, simulator, summary_stat, epsilon=0.5, Nsims=1000):
        # observed data summary statistic
        obs_sumstat = summary_stat(obs_data)
        # draw thetas from prior
        thetas = prior.rvs(size=Nsims)
        # simulated data given a realization of drawn theta
        sim_data = [simulator(theta) for theta in thetas]
        # summary stat of simulated data
        sim_sumstats = [summary_stat(sim) for sim in sim_data]
        # rejection sampler
        samples = [[thetas[i]] for i, sim_sumstat in enumerate(
            sim_sumstats) if distance(sim_sumstat, obs_sumstat) < epsilon]
        # compute acceptance ratio
        accept_ratio = float(len(samples)) / Nsims
        # kernel density estimation of the approximate posterior
        kernel = gaussian_kde(np.array(samples).T[0])

        return kernel, accept_ratio

    def rejection_abc(obs_data, prior, simulator, summary_stat, epsilon=0.5, Nsims=1000):

        posterior_samples = []
        distances = []
        obs_sumstat = summary_stat(obs_data)  # observed data summary statistic

        for i in range(Nsims):
            # draw theta from prior
            theta = prior.rvs(size=1)
            # simulated data given a realization of drawn theta
            sim_data = simulator(theta)
            # simulated data summary statistic
            sim_sumstat = summary_stat(sim_data)
            # keep or discard simulation
            if distance(sim_sumstat, obs_sumstat) < epsilon:
                samples.append(theta)

        # compute acceptance ratio
        accept_ratio = float(len(samples)) / Nsims
        # kernel density estimation of the approximate posterior
        kernel = gaussian_kde(np.array(samples).T[0])

        return kernel, accept_ratio

    def sample(self, observations, n_samples, n_samples_per_param, epsilon, full_output=0):
        """
        Samples from the posterior distribution of the model parameter given the observed
        data observations.
        Parameters
        ----------
        observations: list
            A list, containing lists describing the observed data sets
        n_samples: integer
            Number of samples to generate
        n_samples_per_param: integer
            Number of data points in each simulated data set.
        epsilon: float
            Value of threshold
        full_output: integer, optional
            If full_output==1, intermediate results are included in output journal.
            The default value is 0, meaning the intermediate results are not saved.
        Returns
        -------
        abcpy.output.Journal
            a journal containing simulation results, metadata and optionally intermediate results.
        """

        self.accepted_parameters_manager.broadcast(self.backend, observations)

        self.n_samples = n_samples
        self.n_samples_per_param = n_samples_per_param
        self.epsilon = epsilon

        journal = Journal(full_output)
        journal.configuration["n_samples"] = self.n_samples
        journal.configuration["n_samples_per_param"] = self.n_samples_per_param
        journal.configuration["epsilon"] = self.epsilon

        accepted_parameters = None

        # main Rejection ABC algorithm
        seed_arr = self.rng.randint(
            1, n_samples * n_samples, size=n_samples, dtype=np.int32)
        rng_arr = np.array([np.random.RandomState(seed) for seed in seed_arr])
        rng_pds = self.backend.parallelize(rng_arr)

        accepted_parameters_distances_counter_pds = self.backend.map(
            self._sample_parameter, rng_pds)
        accepted_parameters_distances_counter = self.backend.collect(
            accepted_parameters_distances_counter_pds)
        accepted_parameters, distances, counter = [
            list(t) for t in zip(*accepted_parameters_distances_counter)]

        for count in counter:
            self.simulation_counter += count

        distances = np.array(distances)

        self.accepted_parameters_manager.update_broadcast(
            self.backend, accepted_parameters=accepted_parameters)
        journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
        journal.add_weights(np.ones((n_samples, 1)))
        journal.add_ESS_estimate(np.ones((n_samples, 1)))
        journal.add_distances(copy.deepcopy(distances))
        self.accepted_parameters_manager.update_broadcast(
            self.backend, accepted_parameters=accepted_parameters)
        names_and_parameters = self._get_names_and_parameters()
        journal.add_user_parameters(names_and_parameters)
        journal.number_of_simulations.append(self.simulation_counter)

        return journal
    '''


"""
self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)
journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
journal.add_weights(np.ones((n_samples, 1)))
journal.add_ESS_estimate(np.ones((n_samples, 1)))
journal.add_distances(copy.deepcopy(distances))
self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters)
names_and_parameters = self._get_names_and_parameters()
journal.add_user_parameters(names_and_parameters)
journal.number_of_simulations.append(self.simulation_counter)

###
if journal_file is None:
    journal = Journal(full_output)
    journal.configuration["type_model"] = [type(model).__name__ for model in self.model]
    journal.configuration["type_dist_func"] = type(self.distance).__name__
    journal.configuration["n_samples"] = self.n_samples
    journal.configuration["n_samples_per_param"] = self.n_samples_per_param
    journal.configuration["steps"] = steps
    journal.configuration["epsilon_percentile"] = epsilon_percentile
else:
    journal = Journal.fromFile(journal_file)

if (full_output == 1 and aStep <= steps - 1) or (full_output == 0 and aStep == steps - 1):
        journal.add_accepted_parameters(copy.deepcopy(accepted_parameters))
        journal.add_distances(copy.deepcopy(distances))
        journal.add_weights(copy.deepcopy(accepted_weights))
        journal.add_ESS_estimate(accepted_weights)
        self.accepted_parameters_manager.update_broadcast(self.backend, accepted_parameters=accepted_parameters,
                                                          accepted_weights=accepted_weights)
        names_and_parameters = self._get_names_and_parameters()
        journal.add_user_parameters(names_and_parameters)
        journal.number_of_simulations.append(self.simulation_counter)
"""

if __name__ == "__main__":
    a = 1
    # this returns a dict whose keys are parameter names
    #params = journal.get_parameters()
    #print("Number of posterior samples: {}".format(len(params['mu'])))
    #print("10 posterior samples for mu:")
    # print(params['mu'][0:10])
