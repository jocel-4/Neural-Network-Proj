import numpy as np
from nnCostFunction import nnCostFunction

def numericalGradient(theta, layout, X, y, lam):
    E = 10**-6
    gradApprox = np.zeros((len(theta),1))
    for i in range(len(theta)):
        plus = theta.copy()
        plus[i] += E
        plusVal,_ = nnCostFunction(plus, layout, X, y, lam)
        minus = theta.copy()
        minus[i] -= E
        minusVal,grad = nnCostFunction(minus, layout, X, y, lam)
        gradApprox[i] = (plusVal - minusVal)/(2*E)
    print(np.hstack((grad, gradApprox)))

if __name__ == '__main__':
    params = np.genfromtxt('params.txt').reshape(-1,1)
    layout = [2,4,8,4,4]
    X = np.genfromtxt('X.txt')
    y = np.genfromtxt('y.txt').reshape(-1,1)
    lam = 1.5
    numericalGradient(params, layout, X, y, lam)
