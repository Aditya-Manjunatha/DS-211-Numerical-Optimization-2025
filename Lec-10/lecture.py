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

def backtracking_line_search(f, grad_f, x, p, alpha=0.3, beta=0.8):
    t = 1.
    while f(x + t * p) > f(x) + alpha * t * np.dot(grad_f(x), p):
        t *= beta
    return t

############################################################################################################
# TASK 1:- 
# Implement SR1 update for BFGS
############################################################################################################

# def sr1_update(B, s, y):
#     Bs = B @ s
#     diff = y - Bs
#     denom = np.dot(diff, s)
#     if abs(denom) > 1e-8:  # Avoid division by zero
#         B += np.outer(diff, diff) / denom
#     return B

def sr1_update(B, s, y):
    """
    Symmetric Rank-1 (SR1) update for Hessian approximation.
    """
    Bs = B @ s
    diff = y - Bs
    denom = np.dot(diff, s)

    if abs(denom) > 1e-8 and np.linalg.norm(diff) > 1e-8:  # Avoid division by near-zero
        B += np.outer(diff, diff) / denom  # SR1 update

    return B

def bfgs_update(B, s, y):
    ys = np.dot(y, s)
    if ys == 0:
        return B
    Bs = B @ s
    B -= np.outer(Bs, Bs) / np.dot(s, Bs)
    B += np.outer(y, y) / ys
    return B

# def quasi_newton_sr1(f, grad_f, x0, max_iter=1000, tol=1e-6):
#     x = x0.copy()
#     B = np.eye(len(x0))  # Initial Hessian approximation
#     path = [x.copy()]
    
#     for i in range(max_iter):
#         grad = grad_f(x)
#         direction = -np.linalg.solve(B, grad)  # Stable inverse operation
#         t = backtracking_line_search(f, grad_f, x, direction)
#         x_new = x + t * direction
#         s = x_new - x
#         y = grad_f(x_new) - grad
#         B = sr1_update(B, s, y)
#         x = x_new
#         path.append(x.copy())
        
#         if np.linalg.norm(grad) < tol:
#             break
    
#     return x, i, np.array(path)

def quasi_newton_sr1(f, grad_f, x0, max_iter=1000, tol=1e-6):
    """
    Quasi-Newton optimization using SR1 update.
    """
    x = x0
    path = [x]
    B = np.eye(2)
    for i in range(max_iter):
        grad = rosenbrock_grad(x)
        if np.linalg.norm(grad) < tol:
            break
        p = -np.dot(B, grad)
        alpha = backtracking_line_search(rosenbrock, rosenbrock_grad, x, p)
        s = alpha * p
        x_new = x + s
        y = rosenbrock_grad(x_new) - grad

        Bs = np.dot(B, y)
        diff = s - Bs
        denom = np.dot(diff, y)

        if abs(denom) > 1e-8 and np.linalg.norm(diff) > 1e-8:
            B = B + np.outer(diff, diff) / denom

        x = x_new
        path.append(x)
    return x, i, np.array(path)

def quasi_newton_bfgs(f, grad_f, x0, max_iter=1000, tol=1e-6):
    x = x0.copy()
    B = np.eye(len(x0))  # Initial Hessian approximation
    path = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        direction = -np.linalg.solve(B, grad)
        t = backtracking_line_search(f, grad_f, x, direction)
        x_new = x + t * direction
        s = x_new - x
        y = grad_f(x_new) - grad
        B = bfgs_update(B, s, y)
        x = x_new
        path.append(x.copy())
        
        if np.linalg.norm(grad) < tol:
            break
            
    return x, i, np.array(path)

############################################################################################################
# TASK 2:-
# Plot the paths and solutions for the Rosenbrock function using SR1 and BFGS
############################################################################################################

def plot_sr1_bfgs():
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock((X, Y))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Run optimization methods
    x0 = np.array([-1, 1.5])  # Starting point
    x_sr1, i, sr1_path = quasi_newton_sr1(rosenbrock, rosenbrock_grad, x0)
    x_bfgs, j, bfgs_path = quasi_newton_bfgs(rosenbrock, rosenbrock_grad, x0)
    
    # Subplot 1: SR1
    ax = axes[0]
    ax.contourf(X, Y, Z, levels=np.arange(0, 100, 10), cmap='Blues')
    ax.plot(sr1_path[:, 0], sr1_path[:, 1], 'kx--', markersize=5, label="SR1 Path")
    ax.plot(sr1_path[-1, 0], sr1_path[-1, 1], 'r*', markersize=4, label="Final SR1 Point")
    ax.set_title("SR1 Optimization")
    ax.legend()

    # Subplot 2: BFGS
    ax = axes[1]
    ax.contourf(X, Y, Z, levels=np.arange(0, 100, 10), cmap='Blues')
    ax.plot(bfgs_path[:, 0], bfgs_path[:, 1], 'kx--', markersize=5, label="BFGS Path")
    ax.plot(bfgs_path[-1, 0], bfgs_path[-1, 1], 'r*', markersize=4, label="Final BFGS Point")
    ax.set_title("BFGS Optimization")
    ax.legend()

    plt.tight_layout()
    plt.savefig("sr1_bfgs_paths.png")
    plt.show()

    print("SR1 converged to", x_sr1, "after", i, "iterations")
    print("BFGS converged to", x_bfgs, "after", j, "iterations")

############################################################################################################
# Output:-
plot_sr1_bfgs()
############################################################################################################
