import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize': 'large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.fontsize': 'large',
          'legend.handlelength': 2}
plt.rcParams.update(params)
plt.rc('text', usetex=True)

rng = default_rng()

N = 100
data = []
for _ in range(20):
    data.append(rng.uniform(0, 1, N))

mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

fig = plt.figure(figsize=(6, 4), tight_layout=True, dpi=120)

plt.plot(mean)
plt.fill_between(range(N), mean - std, mean + std,
                 alpha=0.5, facecolor='lightblue')
plt.xlabel("Iterations")
plt.ylabel("Data")
plt.title("Fill Between")
plt.show()
