import numpy as np
import matplotlib.pyplot as plt

# Create a plot of the sign function
fig, ax = plt.subplots(tight_layout=True)
ax.plot([-0.5, 0, 0, 0.5], [-1, -1, 1, 1])
ax.set(xlim=(-0.5, 0.5), title='Sign Function', xlabel='Signal', ylabel='Output')
fig.savefig('../figures/Perceptron_Sign.png')
fig.show()

# Visualize how the decision boundary works
x_values = np.linspace(-1, 1, 101)
boundary = 0.25 - 0.5 * x_values
fig, ax = plt.subplots(tight_layout=True)
ax.plot(x_values, boundary, color='black', label='Boundary')
ax.fill_between(x_values, boundary, 1, alpha=0.8, label='Positive Class')
ax.fill_between(x_values, boundary, -1, alpha=0.8, label='Negative Class')
ax.set(xlim=(-1, 1), ylim=(-1, 1), title='Decision Boundary')
ax.legend()
fig.savefig('../figures/Perceptron_Boundary.png')
fig.show()
