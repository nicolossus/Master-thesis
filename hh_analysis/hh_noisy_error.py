import matplotlib.pyplot as plt
import neuromodels as nm
import numpy as np
import pandas as pd
import pylfi
from scipy import signal


def compute_rmspe(y_true, y_pred):
    """Root mean square percentage error (RMSPE)"""
    rmspe = np.sqrt(
        np.mean(
            np.square(
                (y_true - y_pred) / y_true)
        )
    )
    return rmspe * 100


# Simulator model
T = 120           # Simulation time [ms]
dt = 0.025        # Time step [ms]
I_amp = 10        # Input stimuls amplitude [microA/cm^2]
t_stim_on = 10    # Stimulus onset [ms]
t_stim_off = 110  # Stimulus offset [ms]

stimulus = nm.stimulus.Constant(I_amp, t_stim_on, t_stim_off)

noise_scale = 1.
stimulus_noisy = nm.stimulus.NoisyConstant(
    I_amp, t_stim_on, t_stim_off, noise_scale=noise_scale)

hh = nm.models.HodgkinHuxley(stimulus, T, dt)
hh_noisy = nm.models.HodgkinHuxley(stimulus_noisy, T, dt)

# Observed data
gbarK0 = 36.
gbarNa0 = 120.

V, t = hh(gbar_K=gbarK0,
          gbar_Na=gbarNa0,
          )

V_noisy, t = hh_noisy(gbar_K=gbarK0,
                      gbar_Na=gbarNa0,
                      # noise=True,
                      # noise_scale=0.1,
                      # noise_seed=42
                      )

#
'''
corr = signal.correlate(V_noisy, V)
lags = signal.correlation_lags(len(V), len(V_noisy))
corr /= np.max(corr)

fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, figsize=(4.8, 4.8))
ax_orig.plot(V)
ax_orig.set_title('Original signal')
ax_orig.set_xlabel('Sample Number')
ax_noise.plot(V_noisy)
ax_noise.set_title('Signal with noise')
ax_noise.set_xlabel('Sample Number')
ax_corr.plot(lags, corr)
ax_corr.set_title('Cross-correlated signal')
ax_corr.set_xlabel('Lag')
ax_orig.margins(0, 0.1)
ax_noise.margins(0, 0.1)
ax_corr.margins(0, 0.1)
fig.tight_layout()
plt.show()
'''


rmspe = compute_rmspe(V, V_noisy)
print(rmspe)

plt.plot(t, V_noisy, label='Noisy')
#plt.plot(t, V, label='Original')
plt.legend()
plt.show()
