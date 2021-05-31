
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns

# Set plot style

sns.set()
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": "0.96"})

# Set fontsizes in figures
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'legend.fontsize': 'x-large',
          'font.family': 'serif',
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('text', usetex=True)


# Set path to save the figures
FIGURE_PATH = "./../latex/figures"


def fig_path(fig_id):
    """
    Input name of figure to load or save with extension as dtype str
    """
    return os.path.join(FIGURE_PATH + "/", fig_id)


x = np.linspace(0, 3, 5000)
prior_narrow = stats.beta(2, 4, loc=1, scale=1).pdf(x)
prior_wide = stats.beta(2, 2, loc=0.5, scale=2).pdf(x)

prior_narrow[np.where(prior_narrow == 0)] = np.nan
prior_wide[np.where(prior_wide == 0)] = np.nan


fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True, dpi=800)

ax.plot(x, prior_narrow, lw=1.5, ls='-', color='C0')
ax.fill_between(x, prior_narrow, alpha=0.5, facecolor='lightblue')
ax.plot(x, prior_wide, lw=1.5, ls='-', color='C1')
ax.fill_between(x, prior_wide, alpha=0.5, facecolor='wheat')

ax.annotate('Narrower prior',
            xy=(1.45, 1.5),
            xycoords='data',
            fontsize='x-large',
            xytext=(0.73, 0.6),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black',
                            shrink=0.05,
                            width=2,
                            headwidth=7),
            horizontalalignment='right',
            verticalalignment='top',
            )

ax.annotate('Wider prior',
            xy=(0.72, 0.32),
            xycoords='data',
            fontsize='x-large',
            xytext=(0.28, 0.4),
            textcoords='axes fraction',
            arrowprops=dict(facecolor='black',
                            shrink=0.05,
                            width=2,
                            headwidth=7),
            horizontalalignment='right',
            verticalalignment='top',
            )

ax.set(xlabel=r'$\theta$', ylabel=r'$\pi (\theta)$', xlim=(0, 3))

'''
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')

# remove the ticks from the top and right edges
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
'''
# plt.show()
fig.savefig(fig_path('prior_plot.pdf'), format='pdf', bbox_inches='tight')
