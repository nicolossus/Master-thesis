import matplotlib.pyplot as plt
import numpy as np
import pylfi
import pylfi.priors as priors
import scipy.stats as stats
#from pylfi.inferences import RejectionABC
from pylfi import RejectionABC
from pylfi.distances import DistanceMetrics

# observed data
N = 1000

groundtruth = 2.0
likelihood = stats.norm(loc=0., scale=np.sqrt(groundtruth))
obs_data = likelihood.rvs(size=N)

# simulator model


def gaussian_unkwown_variance(var, n_samples):
    return stats.norm(loc=0., scale=np.sqrt(var)).rvs(size=n_samples)

# summary statistic calculator


def variance(data):
    return np.var(data)


# distance metric
euclidean = DistanceMetrics.euclidean

# initialize sampler
sampler = pylfi.RejectionABC(simulator=gaussian_unkwown_variance,
                             summary_calculator=variance, distance_metric=euclidean)

# priors
#sigma2 = Uniform(name="sigma2")

alpha = 60
beta = 130
sigma2 = priors.InvGamma(
    k=alpha, loc=0, beta=beta, name="sigma2", tex="$\sigma^2$")
priors = [sigma2]

# inference config
n_posterior_samples = 1000
n_simulator_samples_per_parameter = N
epsilon = 0.1

# run inference
journal = sampler.sample(obs_data, priors, n_posterior_samples,
                         n_simulator_samples_per_parameter, epsilon)

# check journal
'''
print(journal.configuration)
print(journal.get_number_of_simulations)
print(journal.get_number_of_accepted_simulations)
print(journal.get_acceptance_ratio)
# print(journal.get_accepted_parameters)
print()
'''

# kernel density estimation of the approximate posterior
samples = journal.get_accepted_parameters["sigma2"]
kernel = stats.gaussian_kde(np.array(samples).T[0])

fraction_accepted = journal.get_acceptance_ratio

a = 1
b = 4
x = np.arange(a, b, 0.01)

point_estimate = np.mean(samples)


alphaprime = alpha + N / 2
betaprime = beta + 0.5 * np.sum(obs_data**2)
posterior = stats.invgamma(alphaprime, loc=0, scale=betaprime)

print(f"groundtruth = {groundtruth}, point estimate = {point_estimate:.3f}")
journal.histplot(true_parameter_values=[2])
