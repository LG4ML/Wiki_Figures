import numpy as np
import matplotlib.pyplot as plt
from typing import List


def add_noise_to_value(points: List[float], norm_scale: float = 0.5):
    return points + np.random.normal(loc=0, scale=norm_scale, size=len(points))


# - - - Figure 1 - - -
# Set x values and calculate noisy y values using linear function
x_values = np.linspace(-5, 5, 50)
y_values = add_noise_to_value(points=x_values*0.5+2, norm_scale=0.8)

# Create best fit line using polyfit
best_fit = np.poly1d(np.polyfit(x=x_values, y=y_values, deg=1))
fit_values = best_fit(x_values)

# Scatter the original points and plot the best fit line
fig, axs = plt.subplots()
axs.scatter(x_values, y_values, color='blue', label='Data Points')
axs.plot(x_values, fit_values, color='red', label='Best Fit Line')
axs.legend()
axs.set(title='Linear Regression', xlim=(min(x_values), max(x_values)))
plt.grid()
fig.savefig(fname=f'../figures/Regression_Linear.png')
plt.show()

# - - - Figure 2 - - -
# Set x values and calculate noisy y values using polynomial function
x_values = np.linspace(-2, 6, 50)
y_values = add_noise_to_value(points=x_values**3 - 6*x_values**2 + 2*x_values - 1.5, norm_scale=2.5)

# Create best fit line using polyfit
best_fit = np.poly1d(np.polyfit(x=x_values, y=y_values, deg=3))
fit_values = best_fit(x_values)

# Scatter the original points and plot the best fit line
fig, axs = plt.subplots()
axs.scatter(x_values, y_values, color='blue', label='Data Points')
axs.plot(x_values, fit_values, color='red', label='Best Fit Line')
axs.legend()
axs.set(title='Polynomial Regression', xlim=(min(x_values), max(x_values)))
plt.grid()
fig.savefig(fname=f'../figures/Regression_Polynomial.png')
plt.show()
