# Import tools for creating the dataset and splitting it later.
from sklearn import datasets, model_selection
# Import plotting tools.
import matplotlib.pyplot as plt
# Import NumPy for later numerical work.
import numpy as np

# Set the number of points to generate.
NUM_DATAPOINTS = 300

# Create two noisy circles. x_data contains each point's two coordinates;
# y_data contains its class label (0 for outside, 1 for inside).
x_data, y_data = datasets.make_circles(n_samples=NUM_DATAPOINTS, noise=0.15, factor=0.6)

# Create a square figure and axes for the plot.
fig, ax = plt.subplots(figsize=(7, 7))
# Plot feature 0 horizontally and feature 1 vertically, coloring points by class.
scatter = ax.scatter(
    x_data[:, 0],
    x_data[:, 1],
    c=y_data,
    edgecolors="grey",
    cmap="Blues",
    vmin=0,
    vmax=1,
)
# Label the axes and identify the two classes in the legend.
ax.set_xlabel("Feature 0")
ax.set_ylabel("Feature 1")
ax.legend(handles=scatter.legend_elements()[0], labels=["outside (0)", "inside (1)"])
# Display the finished plot.
plt.show()
