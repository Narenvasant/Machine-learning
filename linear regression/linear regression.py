from functools import partial

import numpy as np
import matplotlib.pyplot as plt

#from gradient_descent import gradient_descent


def load_dataset(filename="[0]_regression_dataset_1.txt"):
    """Load the given dataset.

    Parameters
    ----------
    filename : string, optional
        Name of the file that contains the data

    Returns
    -------
    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs
    """
    x, y = eval(open(filename, "r").read())
    n_samples = len(x)
    X = np.vstack((np.ones(n_samples), x)).T
    y = np.asarray(y)
    return X, y


def predict(w, X):
    """Predict outputs of a linear model.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    Returns
    -------
    y : array, shape (n_samples,)
        Outputs
    """
    #CODE FOR PREDICT:

    y_hat = np.sum((w * X), axis=1)
    return y_hat


def sse(w, X, y):
    """Compute the sum of squared error of a linear model.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs

    Returns


    -------
    SSE : float
        Sum of squared errors
    """
    
    #CODE FOR SSE:

    y_hat = predict(w, X)
    sse = (y_hat - y)**2
    return np.sum(sse) / 2 

def dSSEdw(w, X, y):
    """Compute the gradient of the sum of squared error.

    Parameters
    ----------
    w : array, shape (n_features + 1,)
        Weights and bias

    X : array, shape (n_samples, n_features + 1)
        Inputs, extended by the "bias feature" which is always 1

    y : array, shape (n_samples,)
        Desired outputs

    Returns
    -------
    g : array, shape (n_features + 1,)
        Sum of squared errors
    """
    #CODE FOR dSSEdW:

    y_hat = predict(w, X)
    diff = (y_hat - y)
    diff = diff[:, np.newaxis]
    dsse = np.sum( diff * X, axis=0)
    return dsse

if __name__ == "__main__":
    X, y = load_dataset()

    # 'partial' creates a new function-like object that only has one argument.
    # The other arguments will contain our dataset.
    grad = partial(dSSEdw, X=X, y=y)

      
    w = np.array([-0.5,0])   #START VALUE
    
    plot_x = []
    plot_w = []
    plot_b = []
    alpha = [0.0001, 0.001, 0.002, 0.0025] #INTIALISING LEARNING RATE
    
    
    #FINDING GRADIENT DESCENT FOR GIVEN LEARNING RATE VALUES, SSE and FINALLY PLOTTING IT:

    for lr in alpha:
        axlegends = []
        plot_x = []
        plot_w = []
        plot_b = []
        plot_sse = []
        w_temp = w.copy()
        for i in range(100):
            plot_x.append(i)
            deriv = grad(w)
            w = w - lr * deriv
            plot_b.append(deriv[0])
            plot_w.append(deriv[1])
            plot_sse.append(sse(w, X, y))
        
        w = w_temp.copy()
        fig = plt.figure()
        ax = plt.subplot(111)
        fig.set_size_inches(20, 10) #Setting figure size
        plt.plot(plot_x, plot_sse)
        ax.set_title('SSE vs Iterations')
        axlegends.append('SSE for Lr= ' + str(lr))
    
        plt.plot(plot_x, [0 for i in range(len(plot_x))], c='g') #FOR ZERO LINE REFERENCE
        axlegends.append('Zero Line')  
        ax.legend(axlegends)
        plt.xlabel('Iterations')
        plt.ylabel('SSE')
        plt.show()
        
        #The SSE(w*) value for each learning rate:
        print('Converged SSE value (SSE(w*)) for Learning Rate:' + str(lr) + ' is : ', plot_sse[-1])



