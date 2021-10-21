import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import pandas as pd

# Simulator model
T = 120           # Simulation time [ms]
dt = 0.025        # Time step [ms]
I_amp = 50        # Input stimuls amplitude [microA/cm^2]
t_stim_on = 10    # Stimulus onset [ms]
t_stim_off = 110  # Stimulus offset [ms]
gbarK0 = 36.
gbarNa0 = 120.

I_amps = np.linspace(7, 50, 51)
isi_mean = []
isi_sd = []

for I_amp in I_amps:
    stimulus = nm.stimulus.Constant(I_amp, t_stim_on, t_stim_off)
    hh = nm.models.HodgkinHuxley(stimulus, T, dt)
    sps = nm.statistics.SpikeStats(t_stim_on, t_stim_off, threshold=-20, rfp=1)
    V, t = hh(gbarK0, gbarNa0)
    spike_data = sps.find_spikes(V, t)
    isi = sps.isi(spike_data['spike_times'])
    isi_mean.append(np.mean(isi))
    isi_sd.append(np.std(isi))


m, b = np.polyfit(isi_mean, isi_sd, 1)

print(f'{m=}')
print(f'{b=}')
rfp_eff = b / m
print(f'{rfp_eff=}')

plt.plot(isi_mean, isi_sd)
plt.show()
