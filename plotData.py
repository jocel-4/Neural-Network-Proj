import numpy as np
import matplotlib.pyplot as plt
from predict import predict

def scatter(X, y, num_labels):
    X = X[:, 1:]
    colors = ['red', 'orange', 'green', 'blue', 'yellow', 'purple']
    for i in range(1, num_labels+1):
        X1 = X[np.where(y==i)[0], 0]
        X2 = X[np.where(y==i)[0], 1]
        plt.scatter(X1, X2, marker=i+7, c=colors[i-1])
    if num_labels == 1:
        X1 = X[np.where(y==0)[0], 0]
        X2 = X[np.where(y==0)[0], 1]
        plt.scatter(X1, X2, marker=9, c=colors[1])
    plt.show()

def plot(X, y, num_labels, Theta):
    og_X = X
    X = X[:, 1:]

    # mesh with contours
    step = 0.05
    X1_min, X1_max = X[:,0].min()-1, X[:,0].max()+1
    X2_min, X2_max = X[:,1].min()-1, X[:,1].max()+1
    X1, X2 = np.meshgrid(np.arange(X1_min, X1_max, step),
                       np.arange(X2_min, X2_max, step))

    # make predictions on every point of the meshgrid
    X_grid = np.c_[np.ones((len(X1.ravel()),)), X1.ravel(), X2.ravel()]
    predictions = predict(Theta, X_grid, num_labels)
    predictions = predictions.reshape(X1.shape)
    plt.contourf(X1, X2, predictions, cmap=plt.cm.Paired)

    # scatter plot
    colors = ['red', 'orange', 'green', 'blue', 'yellow', 'purple']
    for i in range(1, num_labels+1):
        X1 = X[np.where(y==i)[0], 0]
        X2 = X[np.where(y==i)[0], 1]
        plt.scatter(X1, X2, marker=i+7, c=colors[i-1])
    if num_labels == 1:
        X1 = X[np.where(y==0)[0], 0]
        X2 = X[np.where(y==0)[0], 1]
        plt.scatter(X1, X2, marker=9, c=colors[1])
    
    plt.show()
