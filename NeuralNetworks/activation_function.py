import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import activations

# Set activation functions and corresponding titles
act_functions = [activations.linear, activations.relu, activations.sigmoid, activations.tanh]
titles = ['Linear Activation', 'ReLu Activation', 'Sigmoid Activation', 'Tanh Activation']

# Set x range using numpy
x = np.linspace(-5, 5, 200)

# Iterate over activation functions and titles
for act_f, title in zip(act_functions, titles):
    fig, axs = plt.subplots()
    y = act_f(x)
    axs.plot(x, y)
    axs.set(title=title, xlim=(-5, 5), ylim=(min(y)-0.1, max(y)+0.1))
    fig.savefig(fname=f'../figures/{title}.png')
    plt.show()
