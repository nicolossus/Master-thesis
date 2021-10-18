import numpy as np
import pandas as pd

s_stats = ["average_AP_overshoot",
           "spike_rate",
           "average_AP_width",
           "average_AHP_depth",
           "latency_to_first_spike",
           "accommodation_index"]

# Normal prior
df = pd.read_csv('data/sum_stats_priorpred_normal.csv')

weights = []

for stat in s_stats:

    r_gbarK = df["gbarK"].corr(df[stat], method='pearson')
    r2_gbarK = r_gbarK**2

    r_gbarNa = df["gbarNa"].corr(df[stat], method='pearson')
    r2_gbarNa = r_gbarNa**2

    weight = np.mean([r2_gbarK, r2_gbarNa])
    weights.append(weight)

df_norm = pd.DataFrame(data=weights,
                       columns=["Weight"],
                       index=s_stats
                       )


df_norm.to_csv('data/sumstat_weights_normal.csv', index=True)

# Uniform prior
df = pd.read_csv('data/sum_stats_priorpred_normal.csv')

weights = []

for stat in s_stats:

    r_gbarK = df["gbarK"].corr(df[stat], method='pearson')
    r2_gbarK = r_gbarK**2

    r_gbarNa = df["gbarNa"].corr(df[stat], method='pearson')
    r2_gbarNa = r_gbarNa**2

    weight = np.mean([r2_gbarK, r2_gbarNa])
    weights.append(weight)

df_unif = pd.DataFrame(data=weights,
                       columns=["Weight"],
                       index=s_stats
                       )


df_unif.to_csv('data/sumstat_weights_uniform.csv', index=True)
