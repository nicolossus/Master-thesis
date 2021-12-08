import neuromodels as nm

# The simulation parameters needed are the simulation time, time
# step and input stimulus:

T = 50.     # Simulation time [ms]
dt = 0.01   # Time step

# Input stimulus can be provided as either a scalar, array or callable
stimulus = nm.stimulus.Constant(I_amp=10,
                                t_stim_on=10,
                                t_stim_off=40
                                )

# Initialize the Hodgkin-Huxley system; model parameters can either be
# set in the constructor or accessed as class attributes:
hh = nm.solvers.HodgkinHuxleySolver(V_rest=-65)
hh.gbarK = 36.0

# The system is solved by calling the class method `solve`:
hh.solve(stimulus, T, dt, method='RK45')

# The solutions can be accessed as class attributes:
t = hh.t
V = hh.V
n = hh.n
m = hh.m
h = hh.h
