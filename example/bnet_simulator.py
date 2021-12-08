import matplotlib.pyplot as plt
import neuromodels as nm

# Model parameters
order = 2500    # -> NE=10,000 ; NI=2500 ; N_tot=12,500 ; CE=1000 ; CI=250
epsilon = 0.1   # Connection probability
T = 1000        # Simulation time [ms]
N_rec = 20      # Record output from N_rec neurons
n_type = 'exc'  # Record excitatory spike trains
D = 1.5         # Synaptic delay [ms]
J = 0.1         # Excitatory synapse weight [mV]

# NEST settings
threads = 16        # Number of threads to use in simulation
print_time = False  # Print simulated time or not

# Simulator model class constructor:
bnet = nm.models.BrunelNet(order=order,
                           epsilon=epsilon,
                           T=T,
                           N_rec=N_rec,
                           n_type=n_type,
                           D=D,
                           J=J,
                           threads=threads,
                           print_time=print_time,
                           )

# The call method takes the synaptic weight parameters `eta` and `g`
# as arguments and returns the output spike trains:
spiketrains = bnet(eta=2.0, g=4.5)

# The simulator class has methods for post-simulation analysis, e.g.:
bnet.rasterplot_rates()
plt.show()
