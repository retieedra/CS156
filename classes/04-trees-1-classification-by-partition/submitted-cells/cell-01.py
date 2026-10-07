import numpy as np
from sklearn import metrics, inspection, datasets
from matplotlib import pyplot as plt

x_data, y_data = datasets.make_moons(n_samples=500, noise=0.15)
plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data)
plt.show()
