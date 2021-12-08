import neuromodels as nm

# The simulation parameters:
T = 50.          # Simulation time [ms]
dt = 0.01        # Time step
t_stim_on = 10   # Stimulus onset
t_stim_off = 40  # Stimulus offset
stimulus = nm.stimulus.Constant(I_amp=10,
                                t_stim_on=t_stim_on,
                                t_stim_off=t_stim_off
                                )

# Initialize the Hodgkin-Huxley simulator:
hh = nm.models.HodgkinHuxley(stimulus, T, dt)

# Call simulator to solve system for passed conductances:
V, t = hh(gbar_K=36., gbar_Na=120.)

# Create a list of summary statistics to extract:
stats = ["average_AP_overshoot",
         "spike_rate",
         "average_AP_width",
         "average_AHP_depth",
         "latency_to_first_spike",
         "accommodation_index"]

# Initialize spike statistics extraction class; stimulus onset
# and offset as well as statistics to extract must be passed
# to the constructor:
sps = nm.statistics.SpikeStats(t_stim_on=t_stim_on,
                               t_stim_off=t_stim_off,
                               stats=stats,
                               threshold=0  # find only spikes above 0 mV
                               )

# The SpikeStats instance is callable; the voltage trace must be
# passed as argument. The extracted summary statistics are returned:
sum_stats = sps(V, t)
print(sum_stats)
