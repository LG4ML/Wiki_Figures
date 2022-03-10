import numpy as np
import matplotlib.pyplot as plt

x = np.array([3, 3.5])
y = np.array([.75, 0.25])

# Euclidean distance
d = np.sqrt(np.sum((x - y) ** 2))
fig, axs = plt.subplots()
# axs.scatter(x[0], x[1], label='A')
axs.scatter(x[0], x[1], label='A')
axs.scatter(y[0], y[1], label='B')
plt.plot([x[0], y[0]], [x[1], y[1]], label='Euklidische Distanz', color='black')
axs.legend()
axs.annotate(f'd={np.round(d, 2)}', (2.0, 1.75))
axs.set(xlim=(0, 4), ylim=(0, 4), title='Euklidische Distanz')
fig.savefig(f'../figures/Euclidean_Distance.png')
plt.show()

# Manhattan distance
d = np.sum(np.abs((x - y)))
fig, axs = plt.subplots()
# axs.scatter(x[0], x[1], label='A')
axs.scatter(x[0], x[1], label='A')
axs.scatter(y[0], y[1], label='B')
plt.plot([x[0], x[0], y[0]], [x[1], y[1], y[1]], label='Manhattan Distanz', color='black')
axs.legend()
axs.annotate(f'd={np.round(d, 2)}', (1.75, .5))
axs.set(xlim=(0, 4), ylim=(0, 4), title='Manhattan Distanz')
fig.savefig(f'../figures/Manhattan_Distance.png')
plt.show()

# Cosine distance
d = 1 - (np.dot(y, x)) / (np.sqrt(np.dot(y, y)) * np.sqrt(np.dot(x, x)))
fig, axs = plt.subplots()
# axs.scatter(x[0], x[1], label='A')
axs.scatter(x[0], x[1], label='A')
axs.scatter(y[0], y[1], label='B')
plt.plot([0, x[0]], [0, x[1]], color='black', label='Ortsvektoren')
plt.plot([0, y[0]], [0, y[1]], color='black')
axs.legend()
axs.annotate(f'd={np.round(d, 2)}', (.75, .5))
axs.set(xlim=(0, 4), ylim=(0, 4), title='Cosine Distanz')
fig.savefig(f'../figures/Cosine_Distance.png')
plt.show()

# Selection of distance metrics
fig, axs = plt.subplots()
axs.scatter(x[0], x[1], label='A')
axs.scatter(y[0], y[1], label='B')
axs.scatter(1.0, 1.0, label='?', color='gray')
axs.legend()
axs.set(xlim=(0, 4), ylim=(0, 4), title='Auswahl der Distanzmetrik')
fig.savefig(f'../figures/Distance_Metrics_Selection.png')
plt.show()
