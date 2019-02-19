from math import sqrt
import numpy as np
from numpy.linalg import eig, inv
from sklearn.datasets import load_iris

def get_majorized_val(w, diag, X, y, u, C):
    """ Finds the value of the majorized function of the objective function """

    # Calculate margin term
    margin_term = (0.5 * np.dot(w @ diag, w))
    print(f"Margin term is: {margin_term}")

    # Calculate hinge term
    hinge_term = 0
    for i in range(len(set_vers_y)):
        hinge_term += pow(1 - set_vers_y[i] * np.dot(w, set_vers_X[i,:]) + u[i], 2) / (4 * u[i])
    print(f"Hinge term is: {hinge_term}")

    return margin_term + C * hinge_term

# Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target
label_names = iris.target_names
features = iris.feature_names

# Append '1' at end of each pattern
X = np.concatenate((X, np.ones(150).reshape(150,1)), axis=1)
print(f"Modified X: \n{X}\n")

# Plot preprocessing
set_vers_X = X[:100,:]
set_vers_y = y[:100]
for i in range(len(set_vers_y)):
    if set_vers_y[i] == 0:
        set_vers_y[i] = -1

# Set w to be initial least squares solution for Setosa & Versicolor
w = inv(np.transpose(set_vers_X) @ set_vers_X) @ np.transpose(set_vers_X) @ set_vers_y
print(f"Initial weight vector:\n{w}\n")

# Set epsilon
epsilon = pow(10,-8)
print(f"Epsilon: {epsilon}\n")

# Intialize Lagrange multipliers {u}
u = [1 for i in range(len(set_vers_y))]
print(f"Initial u:\n{u}\n")

# Set free parameter C (low values favor margin term over outlier term)
C = 0.1
print(f"Free paramter C: {C}\n")

# Find diagonalized matrix
diag = np.pad(np.identity(len(w)-1), (0,1), 'constant', constant_values=0)

# Initialize list of majorized function values
majorized_func_values = []
majorized_func_values.append(get_majorized_val(w, diag, set_vers_X, set_vers_y, u, C))
print(f"Majorized Function Values: {majorized_func_values}")

while abs(majorized_func_values[len(majorized_func_values)-1] - majorized_func_values[len(majorized_func_values)-2]) < pow(1,-3):

    # Update w
    K = 0
    for j in range(len(set_vers_y)):
        K += np.dot(X[j,:], X[j,:]) / (2 * u[j])
    K = C * K
    print(f"This K: {K}")

    v = 0
    for j in range(len(set_vers_y)):
        v += ((1 + u[j]) * set_vers_y[j] * set_vers_X[j,:]) / (2 * u[j])
    v = C * v
    print(f"This is v: {v}")

    w = inv(diag + K) @ v
    print(f"New w value: {w}\n")

    # Update {u}
    for j in range(len(u)):
        temp = abs(1 - set_vers_y[j] * np.dot(w, set_vers_X[j,:]))
        u[j] = temp if temp > epsilon else epsilon

    # Append new majorized function value
    majorized_func_values.append(get_majorized_val(w, diag, set_vers_X, set_vers_y, u, C))