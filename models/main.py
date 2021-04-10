#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from hodgkin_huxley import HodgkinHuxley
from matplotlib import gridspec
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_prominences, peak_widths
from stimulus import constant_stimulus

# from models.hodgkin_huxley import hh_simulator

T = 120.
dt = 0.01
I_amp = 0.32  # 0.1 #0.31
r_soma = 40  # 15

stimulus = constant_stimulus(
    I_amp=I_amp, T=T, dt=dt, t_stim_on=10, t_stim_off=110, r_soma=r_soma)
# print(stimulus["info"])
I = stimulus["I"]
I_stim = stimulus["I_stim"]

# default parameter values
hh = HodgkinHuxley()
hh.solve(I, T, dt)
t = hh.t
Vm = hh.Vm

# alternative conductances
hh.gbar_K = 32
hh.gbar_Na = 125
hh.solve(I, T, dt)
Vm_sim = hh.Vm

entropy = stats.entropy(Vm, Vm_sim)
print(f"{entropy=}")

w_dist0 = stats.wasserstein_distance(Vm, Vm)
w_dist1 = stats.wasserstein_distance(Vm, Vm_sim)
w_dist2 = stats.wasserstein_distance(Vm_sim, Vm)

print(f"{w_dist0=}")
print(f"{w_dist1=}")
print(f"{w_dist2=}")

'''
voltage trace
'''

'''
# remove top and right axis from plots
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
ax = plt.subplot(gs[0])
plt.plot(t, Vm, lw=2, label='Observed')
plt.plot(t, Vm_sim, '--', lw=2, label='Simulated')
plt.ylabel('Voltage (mV)')
plt.legend()
ax.set_xticks([])
ax.set_yticks([-80, -20, 40])

ax = plt.subplot(gs[1])
plt.plot(t, I_stim, 'k', lw=2)
plt.xlabel('Time (ms)')
plt.ylabel('Stimulus (nA)')

ax.set_xticks([0, np.max(t) / 2, np.max(t)])
ax.set_yticks([0, np.max(I_stim)])
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
plt.show()
'''

'''
find spikes
'''
threshold = -55  # AP threshold
duration = stimulus["duration"]
t_stim_on = stimulus["t_stim_on"]
# peaks, _ = find_peaks(Vm, height=threshold)
peaks, properties = find_peaks(Vm, height=threshold)
V_spikes = properties["peak_heights"]
n_spikes = len(peaks)
spike_rate = n_spikes / duration
average_AP_overshoot = np.sum(V_spikes) / n_spikes
latency_to_first_spike = t[peaks[0]] - t_stim_on
average_AHP_depth = sum([np.min(Vm[peaks[i]:peaks[i + 1]])
                         for i in range(n_spikes - 1)]) / n_spikes


results_half = peak_widths(Vm, peaks, rel_height=0.5)
width_heights, left_ips, right_ips = results_half[1:]

V_idx_arr = np.linspace(0, len(Vm), len(Vm))
V_interpolate = interp1d(x=V_idx_arr, y=Vm)
spike_widths = V_interpolate(right_ips) - V_interpolate(left_ips)
average_AP_width = np.sum(spike_widths) / n_spikes

# for plotting widths
# ------------
t_idx_arr = np.linspace(0, len(t), len(t))
t_interpolate = interp1d(x=t_idx_arr, y=t)
left_ips_physical = t_interpolate(left_ips)
right_ips_physical = t_interpolate(right_ips)
wlines = (width_heights, left_ips_physical, right_ips_physical)
# ------------

#
prominences, left_bases, right_bases = peak_prominences(Vm, peaks)
bases = np.append(left_bases, right_bases[-1])
print("bases")
print(bases)
print(t[bases])
print("argmin")
min_ind = [np.min(Vm[peaks[i]:peaks[i + 1]])
           for i in range(n_spikes - 1)]
print(min_ind)
# print(t[min_ind])
print("left")
print(left_bases)
print(t[left_bases])
print("right")
print(right_bases)
print(t[right_bases])

print(f"{n_spikes=}, {spike_rate=}, {latency_to_first_spike=}, {average_AP_overshoot=}, {average_AHP_depth=}, {average_AP_width=}")


fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
ax = plt.subplot(gs[0])
plt.axhline(threshold, color='r', ls=':', label='Threshold')
plt.hlines(*wlines, color="k", lw=2)
plt.plot(t, Vm, lw=2)
plt.plot(t[peaks], Vm[peaks], "rx", ms=10)
for base in bases:
    plt.axvline(t[base], color='k', ls=':')
plt.ylabel('Voltage (mV)')
plt.legend()
ax.set_xticks([])
ax.set_yticks([-80, -20, 40])

ax = plt.subplot(gs[1])
plt.plot(t, I_stim, 'k', lw=2)
plt.xlabel('Time (ms)')
plt.ylabel('Input (nA)')

ax.set_xticks([0, np.max(t) / 2, np.max(t)])
ax.set_yticks([0, np.max(I_stim)])
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
fig.suptitle("Peaks")
plt.show()


'''
# simulation parameters
T = 120.
dt = 0.01
I_amp = 0.32  # 0.1 #0.31
#I_amp = 0.31
r_soma = 40  # 15

# input stimulus
stimulus = constant_stimulus(
    I_amp=I_amp, T=T, dt=dt, t_stim_on=10, t_stim_off=110, r_soma=r_soma)
# print(stimulus["info"])
I = stimulus["I"]
I_stim = stimulus["I_stim"]

# HH simulation
hh = HodgkinHuxley()
hh.solve(I, T, dt)
t = hh.t
V = hh.V

# parameters for pretty plot
threshold = -55  # AP threshold
t_stim_on = stimulus["t_stim_on"]

# plot voltage trace
fig = plt.figure(figsize=(7, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
ax = plt.subplot(gs[0])
plt.plot(t, V, lw=2)
# , va='center', ha='center')
plt.text(0.0045, 0.25, 'Threshold', fontsize=10,
         color='darkred', transform=plt.gca().transAxes)
plt.axhline(threshold, xmax=0.13, ls='-', color='darkred')
#plt.hlines(threshold, 0, t_stim_on)
plt.ylabel('Voltage (mV)')
ax.set_xticks([])
ax.set_yticks([-80, -20, 40])

ax = plt.subplot(gs[1])
plt.plot(t, I_stim, 'k', lw=2)
plt.xlabel('Time (ms)')
plt.ylabel('Stimulus (nA)')

ax.set_xticks([0, np.max(t) / 2, np.max(t)])
ax.set_yticks([0, np.max(I_stim)])
ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

fig.suptitle("Hodgkin-Huxley Model")
plt.show()
'''
