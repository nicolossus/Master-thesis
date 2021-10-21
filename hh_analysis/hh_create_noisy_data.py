import os

import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import pandas as pd
import pylfi
import seaborn as sns

# Set plot style
sns.set(context="paper", style='darkgrid', rc={"axes.facecolor": "0.96"})

# Set fontsizes in figures
size = 10
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern"],
    "axes.labelsize": size,
    "font.size": size,
    "legend.fontsize": size,
    "xtick.labelsize": size - 1,
    "ytick.labelsize": size - 1,
    'legend.handlelength': 2
}

plt.rcParams.update(tex_fonts)
plt.rc('text', usetex=True)

# Set path to save the figures
FIGURE_PATH = "./../latex/figures"


def fig_path(fig_id):
    """
    Input name of figure to load or save with extension as dtype str
    """
    return os.path.join(FIGURE_PATH + "/", fig_id)


# Simulator model
T = 120           # Simulation time [ms]
dt = 0.025        # Time step [ms]
I_amp = 10        # Input stimuls amplitude [microA/cm^2]
t_stim_on = 10    # Stimulus onset [ms]
t_stim_off = 110  # Stimulus offset [ms]

noise_scale = 1.
stimulus_noisy = nm.stimulus.NoisyConstant(I_amp,
                                           t_stim_on,
                                           t_stim_off,
                                           noise_scale=noise_scale
                                           )

hh_noisy = nm.models.HodgkinHuxley(stimulus_noisy, T, dt)

# Observed data
gbarK0 = 36.
gbarNa0 = 120.

V, t = hh_noisy(gbar_K=gbarK0,
                gbar_Na=gbarNa0,
                )

obs_data = np.stack([V, t], axis=-1)

df = pd.DataFrame(obs_data, columns=["V", "t"])
df.to_csv('data/hh_noisy.csv', index=False)

fig = hh_noisy.plot_voltage_trace(with_stim=True)
fig.savefig(fig_path('hh_noisy_data.pdf'),
            format='pdf',
            dpi=300,
            bbox_inches='tight'
            )
