import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

NUM_DATAPOINTS = 1000

def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))

def generate_data(x):
    a = 3 #beta_1
    b = -5 #beta_0
    prob = sigmoid(a*x+b)
    return np.random.uniform() < prob

x_data = np.random.uniform(high=3, size=NUM_DATAPOINTS)
y_data = [generate_data(x) for x in x_data]
