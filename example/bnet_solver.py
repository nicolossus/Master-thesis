import neuromodels as nm

# Initialize the Brunel network; the `order` parameter determines
# the number of neurons and connections in the network. Model
# parameters can either be set in the constructor or
# accessed as class attributes:
bnet = nm.solvers.BrunelNetworkSolver(order=2500, J=0.35)
bnet.eta = 2.0
bnet.g = 4.5

# The system is solved by calling the class method `simulate`.
# Simulation parameters have default values, but can also be set:
bnet.simulate(T=1000,       # Simulation time [ms]
              dt=0.1,       # Time step
              N_rec=20,     # Number of neurons to record from
              threads=8,    # Number of threads
              )

# The output of the network is returned as `neo.SpikeTrain` objects.
# Whether to return spike trains from excitatory ('exc', default) or
# inhibitory ('inh') neurons is controlled by the `n_type` keyword:
spiketrains = bnet.spiketrains(n_type="exc")

# The `summary` method gives a simple summary of the simulation:
bnet.summary()
