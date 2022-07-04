from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in the Cycling.csv file
data_path = Path('../data/CSV/Cycling.csv')
data = pd.read_csv(data_path, index_col=0)

# Select time window of 3s
data = data[(data['time'] >= 70) & (data['time'] < 73)]
data['z_noisy'] = data['z'] + np.random.normal(loc=0, scale=0.5, size=len(data))
data['z_factor'] = data['z'] * np.random.normal(loc=1, scale=0.25)

fig, axs = plt.subplots(ncols=3, figsize=(24, 6))
fig.suptitle('Data Augmentation for Time Series', fontsize=16)
axs[0].plot(data['time'], data['z'])
axs[0].set(title='Original Sample', xlim=(min(data.time), max(data.time)))
axs[1].plot(data['time'], data['z_noisy'])
axs[1].set(title='Sample with Noise', xlim=(min(data.time), max(data.time)))
axs[2].plot(data['time'], data['z_factor'])
axs[2].set(title='Sample with Factor', xlim=(min(data.time), max(data.time)))
plt.tight_layout()
fig.savefig('../figures/Augmentation.png')
plt.show()
