import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, inv, norm
from sklearn.datasets import load_iris
from sklearn.svm import SVC

def get_hinge_loss_val(w, C, y, vals):
    """ w represents weight vector and bias """
    
    outlier_error = 0
    for i in range(len(y)):
        outlier_error += max(0, 1 - y[i] * vals[i])

    return 0.5 * pow(norm(w), 2) + C * outlier_error

# Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target
label_names = iris.target_names
features = iris.feature_names

# Plot preprocessing
X = np.concatenate((X, np.ones(150).reshape(150,1)), axis=1)
set_vers_X = X[:100,:]
set_vers_y = y[:100]
vers_virg_X = X[50:,:]
vers_virg_y = y[50:]
set_virg_X = np.concatenate((X[:50,:], X[100:,:]), axis=0)
set_virg_y = np.concatenate((y[:50], y[100:]), axis=0)

# Possible values of free parameter C
C_vals = [0.1, 1, 10, 100]

# Setosa & Versicolor Weight Vector, Bias, and Hinge Loss Objective Function Value
for c in C_vals:
    clf = SVC(C=c, gamma='auto', kernel='linear')
    clf.fit(set_vers_X, set_vers_y)
    set_vers_w = clf.coef_[0][:4]
    set_vers_b = clf.coef_[0][4]

    # Get decision function values
    vals = clf.decision_function(set_vers_X)

    # Convert class labels to +1,-1
    for i in range(len(set_vers_y)):
        if set_vers_y[i] == 0:
            set_vers_y[i] = -1

    print(f"=========== Setosa & Versicolor (C = {c}) ===========")
    print(f"Weight Vector: {set_vers_w}")
    print(f"Bias: {set_vers_b}")
    print(f"Hinge Loss Value: {get_hinge_loss_val(clf.coef_[0], c, set_vers_y, vals)}\n")

# Versicolor & Virginica Weight Vector, Bias, and Hinge Loss Objective Function Value
for c in C_vals:
    clf = SVC(C=c, gamma='auto', kernel='linear')
    clf.fit(vers_virg_X, vers_virg_y)
    vers_virg_w = clf.coef_[0][:4]
    vers_virg_b = clf.coef_[0][4]

    # Get decision function values
    vals = clf.decision_function(vers_virg_X)

    # Convert class labels to +1,-1
    for i in range(len(vers_virg_y)):
        if vers_virg_y[i] == 1:
            vers_virg_y[i] = -1
        elif set_virg_y[i] == 2:
            set_virg_y[i] = 1

    print(f"=========== Versicolor & Virginica (C = {c}) ===========")
    print(f"Weight Vector: {vers_virg_w}")
    print(f"Bias: {vers_virg_b}")
    print(f"Hinge Loss Value: {get_hinge_loss_val(clf.coef_[0], c, vers_virg_y, vals)}\n")

# Setosa & Virginica Weight Vector, Bias, and Hinge Loss Objective Function Value
for c in C_vals:
    clf = SVC(C=c, gamma='auto', kernel='linear')
    clf.fit(set_virg_X, set_virg_y)
    set_virg_w = clf.coef_[0][:4]
    set_virg_b = clf.coef_[0][4]

    # Get decision function values
    vals = clf.decision_function(set_virg_X)

    # Convert class labels to +1,-1
    for i in range(len(set_virg_y)):
        if set_virg_y[i] == 0:
            set_virg_y[i] = -1
        elif set_virg_y[i] == 2:
            set_virg_y[i] = 1

    print(f"=========== Setosa & Virginica (C = {c}) ===========")
    print(f"Weight Vector: {set_virg_w}")
    print(f"Bias: {set_virg_b}")
    print(f"Hinge Loss Value: {get_hinge_loss_val(clf.coef_[0], c, set_virg_y, vals)}\n")
