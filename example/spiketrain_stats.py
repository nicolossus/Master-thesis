import neuromodels as nm
import quantities as pq

# Simulator model class constructor:
bnet = nm.models.BrunelNet(order=2500,
                           epsilon=0.1,
                           T=1000,
                           N_rec=20,
                           n_type='exc',
                           D=1.5,
                           J=0.1,
                           threads=16,
                           )

# Simulator call method:
spiketrains = bnet(eta=2.0, g=4.5)

# Create a list of summary statistics to extract:
stats = ["mean_firing_rate",  # rate estimation
         "mean_cv",           # spike interval statistic
         "fanofactor"         # statistic across spike trains
         ]

# Define start and end time as `Quantity` objects:
t_start = 100. * pq.ms  # Cutoff to avoid transient effects
t_stop = 1000 * pq.ms   # End time

# Initialize spike train statistics extraction class;
# start and end time as well as statistics to extract
# must be passed to the constructor:
sts = nm.statistics.SpikeTrainStats(t_start=t_start,
                                    t_stop=t_stop,
                                    stats=stats
                                    )

# The SpikeTrainStats instance is callable; the spike trains must be
# passed as argument. The extracted summary statistics are returned:
sum_stats = sts(spiketrains)

print(sum_stats)
