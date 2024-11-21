# predict whether or not there will be a delay based on kaggle airline delay data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from nnGradientDescent import nnGradientDescent
from predict import predict
import matplotlib.pyplot as plt
from plotData import plot

# load and reform data
data = pd.read_csv('Airlines.csv')
X = data.iloc[:, 1:-1].to_numpy()
y = data['Delay'].to_numpy().reshape(-1,1).astype('int32')

# convert strings to numbers
airlines = list(set(X[:,0]))
froms = list(set(X[:,2]))
tos = list(set(X[:,3]))
X[:,0] = [airlines.index(i) for i in X[:,0]]
X[:,2] = [froms.index(i) for i in X[:,2]]
X[:,3] = [tos.index(i) for i in X[:,3]]
X = np.hstack((np.ones((len(y),1)), X))

X = X.astype('int32')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.999, random_state=0)
#testsize

# other inputs
lam = 0
alpha = 0.1
num_iters = 1000
layout = [7, 16, 1]

# compute
Theta, costs = nnGradientDescent(X_train, y_train, lam, alpha, num_iters, layout)
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

# make predictions and find accuracy
train_pred = predict(Thetas, X_train, layout[-1])
print('Accuracy on training set:', np.count_nonzero(train_pred==y_train)/len(y_train), end='\n\n')
test_pred = predict(Thetas, X_test, layout[-1])
print('Accuracy on test set:', np.count_nonzero(test_pred==y_test)/len(y_test), end='\n\n')


# plot data
#plot(X_train, y_train, num_labels, Theta1, Theta2)
plt.plot(range(num_iters), costs)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.show()
