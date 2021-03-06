#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from hodgkin_huxley import HodgkinHuxley
from matplotlib import gridspec
from spiking_features import SpikingFeatures
from stimulus import constant_stimulus, equilibrating_stimulus

# plt.style.use('seaborn')
sns.set()
sns.set_context("paper")
sns.set_style("darkgrid", {"axes.facecolor": "0.96"})

# Set fontsizes in figures
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large',
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('text', usetex=True)

# remove top and right axis from plots
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# simulation parameters
T = 200
dt = 0.025
I_amp = 0.32
r_soma = 40
threshold = -55  # AP threshold

# input stimulus
stimulus = equilibrating_stimulus(
    I_amp=I_amp, T=T, dt=dt, t_stim_on=[5, 50], t_stim_off=[15, 150], r_soma=r_soma)
# print(stimulus["info"])
I = stimulus["I"]
I_stim = stimulus["I_stim"]

# HH simulation
hh = HodgkinHuxley()
hh.solve(I, T, dt)
t = hh.t
V = hh.V
n = hh.n
m = hh.m
h = hh.h

# plot voltage trace
fig = plt.figure(figsize=(8, 7), tight_layout=True, dpi=120)
gs = gridspec.GridSpec(3, 1, height_ratios=[4, 4, 1])
ax = plt.subplot(gs[0])
plt.plot(t, V, lw=1.4)
plt.axhline(hh.V_rest, ls=':', color='r')
plt.ylabel('Voltage (mV)')
ax.set_xticks([])
ax.set_yticks([-80, -55, -20, 10, 40])

ax = plt.subplot(gs[1])
plt.plot(t, n, '-', lw=1.4, label='$n$')
plt.plot(t, m, "-", lw=1.4, label='$m$')
plt.plot(t, h, ls='-', lw=1.4, label='$h$')
plt.legend(loc='upper right')
plt.ylabel("State")
ax.set_xticks([])
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

ax = plt.subplot(gs[2])
plt.plot(t, I_stim, 'k', lw=1.4)
plt.xlabel('Time (ms)')
plt.ylabel('Stimulus (nA)')

#ax.set_xticks([0, 10, 25, 40, np.max(t)])
ax.set_yticks([0, np.max(I_stim)])
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

fig.suptitle("Hodgkin-Huxley Model")
plt.show()
