import matplotlib.pyplot as plt
import neuromodels as nm

# The simulation parameters needed are the simulation time,
# time step and input stimulus:
T = 50.     # Simulation time [ms]
dt = 0.01   # Time step
stimulus = nm.stimulus.Constant(I_amp=10,
                                t_stim_on=10,
                                t_stim_off=40
                                )

# Initialize the Hodgkin-Huxley simulator; simulation and fixed
# model parameters are passed to the constructor
hh = nm.models.HodgkinHuxley(stimulus,
                             T,
                             dt,
                             method='RK45',     # integration method
                             pdict={},          # dict of model params
                             solver_options={}  # dict of solver opts
                             )

# Calling the instance solves the HH system for the passed values
# of the active conductances, and the voltage trace is returned
V, t = hh(gbar_K=36., gbar_Na=120.)

# The simulator class has methods for post-simulation analysis
hh.plot_voltage_trace(with_stim=True)
plt.show()
