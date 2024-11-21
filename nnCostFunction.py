import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

# make sure to initilize randomly per layer
# order = 'F' has been removed
def nnCostFunction(params, layout, X, y, lam):
    #initializations
    m = len(y)
    num_labels = layout[-1]
    params = params.copy()
    layers = len(layout)

    # transform vector to matrices
    start = 0
    Theta = []
    for i in range(layers-1):
        end = start+(layout[i]+1)*layout[i+1]
        Theta.append(params[start:end].reshape(layout[i+1], layout[i]+1, order='F'))
        start = end

    # forward propogation : make predictions
    a = X
    zz = []
    aa = [X]
    for theta in Theta:
        z = np.matmul(a, theta.T)
        a = np.hstack((np.ones((m,1)), sigmoid(z)))
        zz.append(z)
        aa.append(a)
    a = a[:, 1:]

    # calculate cost
    y = (np.arange(1, num_labels+1) == y).astype(int)
    lamTerm = lam/(2*m) * np.sum([np.sum(t[:,1:]**2) for t in Theta])
    J = -1/m * np.sum(y*np.log(a) + (1-y)*np.log(1-a)) + lamTerm

    # back propogation : get gradients
    prevDelta = a - y
    deltas = [prevDelta]
    for i in range(layers-2, 0, -1):
        temp = np.matmul(deltas[0], Theta[i]) # 2, 1 or 1
        temp = temp * np.hstack((np.ones((m,1)), sigmoidGradient(zz[i-1]))) # 1,0 or 0
        deltas.insert(0, temp[:, 1:])

    big_deltas = [np.matmul(deltas[i].T, aa[i]) for i in range(len(deltas))]
    for theta in Theta:
        theta[:, 0] = np.zeros((theta.shape[0],))

    grad = np.array([[0]]) # just creating array so it can be staccked
    for i in range(len(big_deltas)):
        temp = (1/m * (big_deltas[i] + lam * Theta[i])).reshape(-1, 1, order='F')
        grad = np.vstack((grad, temp))
    grad = grad[1:] # taking out the extra 0

    return J, grad


if __name__ == '__main__':
    # load data
    X = np.genfromtxt('X.txt')
    y = np.genfromtxt('y.txt').reshape(-1,1)
    params = np.genfromtxt('params.txt').reshape(-1,1)
    #params = np.random.randn(108,1)
    layout = [2,4,8,4,4]
    lam = 1.5 # 12 + 20

    # compute
    J, grad = nnCostFunction(params, layout, X, y, lam)
    print('Cost:', J)
    print('Gradients:', grad)
