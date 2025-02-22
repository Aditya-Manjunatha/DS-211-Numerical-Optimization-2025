# What are the coditions on the eigen values of a hessian to be invertible ?

# In newton step, Hessian may not be invertible as it need not be PD.
# FULL Netwton step is delta = - H^-1 * g
# If you dont want full newton step, you multiply by a scalar alpha and use backtracking line search from alpha = 1 to 0 to find the best alpha
# delta = - alpha * H^-1 * g

# Want an iterative scheme to find the hessian inverse 

# Let B_k be the approximation to the inverse of the hessian at kth iteration

"""
Want a algorithm like 
loop over k :
g_k = grad(f(x_k))
B_k = ...
x_k+1 = x_k + @ * B_k * g_k

# One idea is the rank 1 update 
B_k+1 = B_k + gamma * U * U^T
Where U is a column vector and gamma is a scalar

How to find U and gamma ?

We know that H_k+1 = grad(f(x_k+1)) - grad(f(x_k)) / x_k+1 - x_k
Let S_k = x_k+1 - x_k
Let Y_k = grad(f(x_k+1)) - grad(f(x_k))

Then H_k+1 * S_k = Y_k

Thus S_k = B_k * Y_k

"""

############################################################################################################
# TASK 1 :-
# Define Rosenbrock function, its jacobian and hessian, visualize contours (a=1., b=100)
############################################################################################################

import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, a=1., b=100.):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x, a=1., b=100.):
    dx = 2 * (x[0]-1) + 400 * x[0] * (x[0]**2 - x[1])
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

def rosenbrock_hessian(x, a=1., b=100.):
    dxx = 2 + 400 * (3 * x[0]**2 - x[1])
    dxy = -400 * x[0]
    dyx = -400 * x[0]
    dyy = 200
    return np.array([[dxx, dxy], [dyx, dyy]])

def plot_rosenbrock():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock((X, Y))
    plt.contourf(X, Y, Z, levels = np.arange(0, 100, 10), cmap='Blues')
    plt.colorbar()
    plt.savefig('rosenbrock.png')
    plt.show()


############################################################################################################
# TASK 2 :-
# Recall backtracking line search function for step length selection
############################################################################################################

def backtracking_line_search(f, grad_f, x, p, alpha=0.3, beta=0.8):
    t = 1.
    while f(x + t * p) > f(x) + alpha * t * np.dot(grad_f(x), p):
        t *= beta
    return t

############################################################################################################
# TASK 3 :-
# Apply gradient descent with backtracking step length selection

def gradient_descent(f, x_0, tol = 1e-6, max_iter = 3000):
    x = x_0
    path = [x]
    for i in range(max_iter):
        grad = rosenbrock_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        p = -grad
        alpha = backtracking_line_search(rosenbrock, rosenbrock_grad, x, p)
        x = x + alpha * p
        path.append(x)
    return x, i, path

############################################################################################################
# TASK 4 :-
# Apply Newton's method with backtracking line search
############################################################################################################

def newton_method(f, grad_f, hessian_f, x_0, tol = 1e-6, max_iter = 1000):
    x = x_0
    path = [x]
    for i in range(max_iter):
        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break
        hessian = hessian_f(x)
        # Descent direction in newton method
        p = np.linalg.solve(hessian, -grad)
        alpha = backtracking_line_search(f, grad_f, x, p)
        x = x + alpha * p
        path.append(x)
    return x, i, path

############################################################################################################
# TASK 5 :-
# Plot the contour of the Rosenbrock function and the path taken by gradient descent and Newton's method
############################################################################################################

def plot_rosenbrock_with_path():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock((X, Y))
    plt.contourf(X, Y, Z, levels = np.arange(0, 100, 10), cmap='Blues')
    plt.colorbar()
    x_gd, no_iter_gd, gd_path = gradient_descent(rosenbrock, np.array([0.6, 0.6]))
    x_newton, no_iter_newton, newton_path = newton_method(rosenbrock, rosenbrock_grad, rosenbrock_hessian, np.array([-1, 1]))

    # Plot the solution in red and the path in black
    plt.scatter(x_gd[0], x_gd[1], color='red', marker='x')
    plt.scatter(x_newton[0], x_newton
                [1], color='green', marker='x') 
    
    gd_path = np.array(gd_path)
    newton_path = np.array(newton_path)
    # plt.plot(gd_path[:, 0], gd_path[:, 1], 'r--', label="Gradient Descent")  # Red dashed line
    # plt.plot(gd_path[:, 0], gd_path[:, 1], 'ko', markersize=4)  # Black circles at each point

    plt.plot(gd_path[:, 0], gd_path[:, 1], 'kx', markersize=5, label="Gradient Descent Steps")

# Mark the final solution with a red 'X'
    plt.plot(gd_path[-1, 0], gd_path[-1, 1], 'rx', markersize=8, markeredgewidth=2, label="Final Solution")

    #print(gd_path[:10])

    # Plot Newton's Method path (green dashed line)
    plt.plot(newton_path[:, 0], newton_path[:, 1], 'g--', label="Newton Method")
    plt.legend(['Gradient Descent', 'Newton Method'])
    plt.savefig('rosenbrock_path.png')
    plt.show()
    print(f"Gradient Descent RED Plot: {x_gd}, {no_iter_gd}")
    print(f"Newton Method GREEN Plot: {x_newton}, {no_iter_newton}")

############################################################################################################
plot_rosenbrock_with_path()
"""
Output :-
Gradient Descent RED Plot: [0.9876785  0.97549659], 2999
Newton Method GREEN Plot: [1. 1.], 19
"""
############################################################################################################

"""
BUT ISSUE IS THAT WHAT IF WE DONT HAVE HESSIAN INVERSE ?
OR CANT WRITE A CLOSED FORM EXP LIKE WE DID FOR ROSENBROCK ?

WE NEED APPROXIMATIONS TO HESSIAN INVERSE WHICH ARE FOUND ITERATIVELY

QUASI NEWTON METHODS WILL BE EXPLROED IN NEXT LECTURE
"""