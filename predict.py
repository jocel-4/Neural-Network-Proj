# predict given multiple thetas
import numpy as np
from sigmoid import sigmoid

def predict(Theta, X, num_labels=2): # theta is a list of Thetas per layer
    m = X.shape[0]
    a = X
    
    for theta in Theta:
        z = np.matmul(a, theta.T)
        a = np.hstack((np.ones((m,1)), sigmoid(z)))

    a = a[:, 1:]
    if num_labels == 1:
        return np.round(a)
    return (np.argmax(a, axis=1)+1).reshape(m,1)
