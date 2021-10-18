import numpy as np
import pandas as pd

s_stats = ["average_AP_overshoot",
           "spike_rate",
           "average_AP_width",
           "average_AHP_depth",
           "latency_to_first_spike",
           "accommodation_index"]

headers = s_stats + ["epsilon", "n_sims"]

results = []

n_sims_lst = [100, 200]
for n_sims in n_sims_lst:
    stat_scales = [1., 2, 3, 4, 5, 6]
    epsilon = 0.5
    results.append([*stat_scales, epsilon, n_sims])


data = dict(zip(headers, np.stack(results, axis=-1)))
df = pd.DataFrame.from_dict(data)
