import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def generate_cluster_from_center(center_point: Tuple[float, float], n_points: int = 50,
                                 norm_scale: float = 0.5) -> List[Tuple[float, float]]:
    return [(center_point[0] + np.random.normal(0, norm_scale), center_point[1] + np.random.normal(0, norm_scale))
            for _ in range(n_points)]


def add_noise_to_points(points: List[Tuple[float, float]], norm_scale: float = 0.05):
    return [p[0] + np.random.normal(0, norm_scale) for p in points], \
           [p[1] + np.random.normal(0, norm_scale) for p in points]


# - - - Figure 1 - - -
# Set different centers
centers = [(0.5, 0.5), (2.5, 3.0), (4.0, 0.5)]
labels = ['A', 'B', 'C']

# Create figure and itereate over center points
fig, axs = plt.subplots()
for center, label in zip(centers, labels):
    cluster_points = generate_cluster_from_center(center_point=center)
    x, y = [p[0] for p in cluster_points], [p[1] for p in cluster_points]
    axs.scatter(x, y, label=f'Class {label}')
axs.scatter(2.0, 1.5, label='New point', color='black')
axs.legend()
axs.set(title='Multiclass Classification')
fig.savefig(fname=f'../figures/Classification_Multiclass.png')
plt.show()


# - - - Figure 2 - - -
fig, axs = plt.subplots()

# Generate center cluster
cluster = generate_cluster_from_center(center_point=(0.4, 0.4), n_points=100, norm_scale=0.1)
cluster_x, cluster_y = [p[0] for p in cluster], [p[1] for p in cluster]

# Genrate circular cluster around center cluster
circle_x, circle_y = add_noise_to_points(points=[(x, np.sqrt(1 - np.square(x))) for x in np.random.random(100)])

# Plot both clusters
axs.scatter(cluster_x, cluster_y, label='Class A')
axs.scatter(circle_x, circle_y, label='Class B')
axs.legend()
axs.set(title='Binary Classification')
fig.savefig(fname=f'../figures/Classification_Binary.png')
plt.show()
