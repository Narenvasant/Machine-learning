import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_almost_equal

def gradient_descent(x0, alpha, grad, n_iter=100, return_path=False):
    """Gradient descent.

    Parameters
    ----------
    x0 : array-like, shape (n_params,)
        Initial guess for parameter vector that will be optimized

    alpha : float
        Learning rate, should be within (0, 1), typical values are 1e-1, 1e-2,
        1e-3, ...

    grad : callable, array -> array
        Computes the derivative of the objective function with respect to the
        parameter vector

    n_iter : int, optional (default: 100)
        Number of iterations

    return_path : bool, optional (default: False)
        Return the path in parameter space that we took during the optimization

    Returns
    -------
    x : array, shape (n_params,)
        Optimized parameter vector

    path : list of arrays (shape (n_params,)), optional
        Path that we took in parameter space
    """

#IMPLEMENTING GRADIENT DESCENT:

    iterations = n_iter
    if return_path:
        accumulated_x0 = []
        newx0 = x0.copy()
        for i in range(iterations):
            deriv = grad(newx0.copy())
            newx0 = newx0 - alpha * deriv
            accumulated_x0.append(newx0.copy())
        accumulated_x0 = np.array(accumulated_x0)
        return newx0, accumulated_x0
    else:
        newx0 = x0.copy()
        for i in range(iterations):
            deriv  = grad(newx0.copy())
            newx0 = newx0 - alpha * deriv
        return newx0
    
    

#ASSIGNING START VALUE, LEARNING RATE AND FUNCTION DEFINITION:

def test_square():
    def fun_square(x):
        return np.linalg.norm(x) ** 2
    def grad_square(x):
        return 2.0 * x

    random_state = np.random.RandomState(42)
    
    x = gradient_descent(
        x0=random_state.randn(5), #start value
        alpha=0.1,                #learning rate
        grad=grad_square,
        n_iter=100,
        return_path=False
    )
    
    f = fun_square(x)              #function
    assert_almost_equal(f, 0.0)



#PLOTTING THE GENERATED GRADIENT DESCENT:

def plot_square():
    def fun_square(x):
        return np.linalg.norm(x) ** 2
    def grad_square(x):
        return 2.0 * x

    x, path = gradient_descent(
        x0=np.ones(1) ,
        alpha=0.7,
        grad=grad_square,
        n_iter=5,
        return_path=True
    )
    f = fun_square(x)
    plt.plot(path, [fun_square(x) for x in path], "-o",
             label="Gradient Descent Path")
    plt.plot(np.linspace(-1, 1, 1000),
             [fun_square(x) for x in np.linspace(-1, 1, 1000)],
             label="$f(x) = x^2$")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc="best")
    plt.show()

#MAIN PROGRAM TO CALL THE RESPECTIVE FUNCTION PROGRAMS:

if __name__ == "__main__":
    test_square()
    plot_square()






