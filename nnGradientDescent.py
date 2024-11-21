# multi-hidden layer neural network gradient descent
import numpy as np
from nnCostFunction import nnCostFunction
from predict import predict
import matplotlib.pyplot as plt
from plotData import plot

def nnGradientDescent(X, y, lam, alpha, num_iters, layout):
    # initializations
    m = len(y)
    num_labels = layout[-1]
    layers = len(layout)
    J_history = np.zeros((num_iters, 1))

    # randomly generating Theta
    Theta = np.array([[0]])
    for i in range(layers-1):
        E = np.sqrt(2)/np.sqrt(layout[i] + layout[i+1])
        new_theta = np.random.randn((layout[i]+1)*layout[i+1], 1)*E
        Theta = np.vstack((Theta, new_theta))
    Theta = Theta[1:]

    # training model
    for i in range(num_iters):
        J, grad = nnCostFunction(Theta, layout, X, y, lam)
        Theta = Theta - alpha * grad
        J_history[i, 0] = J

    return Theta, J_history
    


if __name__ == '__main__':
    # make sure to try 0, 1, 2, 3 hidden layers
    X = np.genfromtxt('X.txt')
    y = np.genfromtxt('y.txt').reshape(X.shape[0], 1)
    lam = 0
    alpha = 0.3
    num_iters = 50000
    layout = [2,15,15,4]

    # compute
    Theta, costs = nnGradientDescent(X, y, lam, alpha, num_iters, layout)
    print('Cost:', costs[-1], end='\n\n')

    # transform Theta to matrices
    start = 0
    Thetas = []
    for i in range(len(layout)-1):
        end = start + (layout[i]+1)*layout[i+1]
        t = Theta[start:end].reshape(layout[i+1], layout[i]+1, order='F')
        Thetas.append(t)
        start = end
        print('Theta ' + str(i+1) + ': ', t, end='\n\n')

    # find accuracy
    predictions = predict(Thetas, X)
    print('Accuracy:', np.count_nonzero(predictions == y)/len(y), end='\n\n')

    # plot data
    plot(X, y, layout[-1], Thetas)
    plt.plot(range(num_iters),  costs)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.show()
