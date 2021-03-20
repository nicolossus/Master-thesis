import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from kde import KDE
from KDEpy import FFTKDE
from KDEpy.bw_selection import improved_sheather_jones, silvermans_rule

np.random.seed(42)


def kl_div(P, Q):
    """Relative entropy from Q to P

    Usually P represents the data, the observations, or a probability
    distribution precisely measured. Distribution Q represents instead a
    theory, a model, a description or an approximation of P.

    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence


# Generate data
N = 1024
#N = 2048
likelihood = stats.norm(loc=0., scale=np.sqrt(2))
data = likelihood.rvs(size=N)
x = np.linspace(-6, 6, N)
true_pdf = likelihood.pdf(x)

# sklearn kde
kde = KDE(data, bandwidth='auto', kernel='gaussian')
density = kde(x)
best_params = kde.best_params
print('Optimal params sklearn:', best_params)

kde2 = KDE(data, bandwidth=2.0, kernel='gaussian')
density2 = kde2(x)

# kdepy kde
# Silverman assumes normality of data - use ISJ with much data instead
silverman_bw = silvermans_rule(data.reshape(-1, 1))  # Shape (obs, dims)
isj_bw = improved_sheather_jones(data.reshape(-1, 1))
print(f"Silverman bandwidth: {silverman_bw}")
print(f"ISJ bandwidth: {isj_bw}")
y1 = FFTKDE(kernel='gaussian', bw="silverman").fit(data).evaluate(x)
y2 = FFTKDE(kernel='gaussian', bw='ISJ').fit(data).evaluate(x)

fig, ax = plt.subplots(1, 1)
ax.plot(x, true_pdf, label='true pdf')
ax.plot(x, density, label='sklearn kde')
ax.plot(x, density2, label='sklearn bad kde')
ax.plot(x, y1, label='kdepy1 silver')
ax.plot(x, y2, label='kdepy2 ISJ')
ax.set_ylabel('Density')
ax.set_xlabel('$x$')
ax.legend()
plt.show()

print(f"KL div from kdepy1 to true pdf: {kl_div(true_pdf, y1):.4f}")
print(f"KL div from kdepy2 to true pdf: {kl_div(true_pdf, y2):.4f}")
print(
    f"KL div from sklearn optimal kde to true pdf: {kl_div(true_pdf, density):.4f}")
print(
    f"KL div from sklearn bad kde to true pdf: {kl_div(true_pdf, density2):.4f}")
