import numpy as np
import matplotlib.pyplot as plt

######################################################################################################################
# First ORDER GRADEINT DESCENT METHOD :- CAN WORK WITH ANY TYPE OF FUNCTIONS
######################################################################################################################

# Create a 2D callable function representing x[0]^2 + 2*x[1]^2 + 0.5*x[0]*x[1]
def f(x):
    return x[0]**2 + x[1]**2 + 0.5*x[0]*x[1]

# Define gradient of f
def grad_f(x):
    return np.array([2*x[0] + 0.5*x[1], 2*x[1] + 0.5*x[0]])


# 1D backtracking line search method
alpha_cap = 1  # Must be positive
rho = 0.2  # Must lie in (0, 1)
c = 1e-4  # Must lie in (0, 1)
x = np.array([5, 5])
descent_direction = -grad_f(x)


def backtracking_line_search(f, x, descent_direction, rho, c, alpha_cap):
    alpha = alpha_cap
    while f(x + alpha*descent_direction) > f(x) + c*alpha*np.dot(grad_f(x), descent_direction):
        alpha = rho*alpha
    return alpha


# Define a function to evaluate whether a given descent direction is a valid descent direction
def is_descent_direction(f, x, descent_direction):
    return np.dot(grad_f(x), descent_direction) < 0


# Perform Gradient Descent :-

def gradient_descent(f, x_0, grad_f, max_iter, tol = 1e-6):
    x = x_0
    path = [x]
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        descent_direction = -grad_f(x)
        alpha = backtracking_line_search(f, x, descent_direction, rho, c, alpha_cap)
        x = x + alpha*descent_direction
        # Save the path of descent :-
        path.append(x)

    return x, f(x), i, path

# Plot the path of descent :-
(x, f_x, i, path) = gradient_descent(f, np.array([5, 5]), grad_f, 1000, 1e-6)

X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(X, Y)

# Look at what is happening below, f was taking a 2D input, now what is it doing ?
Z = f(np.array([X, Y]))

path = np.array(path)
plt.contour(X, Y, Z, levels = 100)
plt.plot(path[:, 0], path[:, 1], 'r-o')
plt.title(f'Path with alpha = {alpha_cap}, rho = {rho}, c = {c}')
plt.show()
plt.savefig(f'path_with_alpha_{alpha_cap}_rho_{rho}_c_{c}.png')
# If you see, the path is very back and forth, why was this the case ?
# Because of the step size, we took alpha_cap and rho = 1, 0.5
print(f"Number of iterations taken with backtracking line search: {i}") # Became 13
print(f"Value of f at the minimum: {f_x}")
print(f"Value of x at the minimum: {x}")

# Lets put alpha_cap = 1, rho = 0.7
# i = 59 !!!!!

# Lets put alpha_cap = 1, rho = 0.9
# i = 86

# Lets put alpha_cap = 1, rho = 0.25
#


######################################################################################################################
# Do gradient descent, with a fixed step size, and see how it performs.
######################################################################################################################


def gradient_descent_fixed_step_size(f, x_0, grad_f, max_iter, step_size, tol = 1e-6):
    x = x_0
    path = [x]
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        x = x - step_size*grad_f(x)
        path.append(x)
    return x, f(x), i, path

x, f_x, i, path = gradient_descent_fixed_step_size(f, np.array([5, 5]), grad_f, 1000, 0.01)

X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(X, Y)

# Look at what is happening below, f was taking a 2D input, now what is it doing ?
Z = f(np.array([X, Y]))

path = np.array(path)
plt.contour(X, Y, Z, levels = 100)
plt.plot(path[:, 0], path[:, 1], 'r-o')
plt.title(f'Path with fixed step size')
plt.show()
plt.savefig(f'path_with_fixed_step_size_with_alpha_{alpha_cap}_rho_{rho}_c_{c}.png')

print(f"Number of iterations taken with out backtracking line search: {i}")
print(f"Value of f at the minimum: {f_x}")
print(f"Value of x at the minimum: {x}")


# Took 660 iterations!!!!
# So we were doing really good with backtracing line search

######################################################################################################################
# Take descent direction to be alternating between -x[0] and -x[1] depending on the iteration number and see how it performs.
######################################################################################################################
def descent_direction_alternating(x, i):
    return -x[0] if i % 2 == 0 else -x[1]

def gradient_descent_alternating(f, x_0, grad_f, max_iter,step_size, tol = 1e-6):
    x = x_0
    path = [x]
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        descent_direction = descent_direction_alternating(x, i)
        x = x + step_size*descent_direction
        path.append(x)
    return x, f(x), i, path

x, f_x, i, path = gradient_descent_alternating(f, np.array([5, 5]), grad_f, 1000, 0.01)

X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(X, Y)
Z = f(np.array([X, Y]))
# Plot the path of descent :-
path = np.array(path)
plt.contour(X, Y, Z, levels = 100)
plt.plot(path[:, 0], path[:, 1], 'r-o')
plt.title(f'Path with alternating descent direction')
plt.show()
plt.savefig(f'path_alternating_direction_with_alpha_{alpha_cap}_rho_{rho}_c_{c}.png')

print(f"Number of iterations taken with alternating descent direction: {i}")
print(f"Value of f at the minimum: {f_x}")
print(f"Value of x at the minimum: {x}")


######################################################################################################################
# Take descent direction to be random and keep checking if it is valid in each iteration
# If valid, descent in that direction.
# If not , then generate a new random direction.
######################################################################################################################
def descent_direction_random(x_0, i):
    return np.random.randn(2)

def gradient_descent_random(f, x_0, grad_f, max_iter,step_size, tol = 1e-6):
    descent_direction = descent_direction_random(x_0, 0)
    x = x_0
    path = [x]
    for i in range(max_iter):
        if np.linalg.norm(grad_f(x)) < tol:
            break
        while is_descent_direction(f, x, descent_direction) == False:
            descent_direction = descent_direction_random(x, i)
        x = x + step_size*descent_direction
        path.append(x)
    return x, f(x), i, path

x, f_x, i, path = gradient_descent_random(f, np.array([5, 5]), grad_f, 10000, 0.01)

X = np.linspace(-10, 10, 100)
Y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(X, Y)
Z = f(np.array([X, Y]))
# Plot the path of descent :-
path = np.array(path)
plt.contour(X, Y, Z, levels = 100)
plt.plot(path[:, 0], path[:, 1], 'r-o')
plt.title(f'Path with random descent direction')
plt.show()
plt.savefig(f'path_random_direction_with_alpha_{alpha_cap}_rho_{rho}_c_{c}.png')

print(f"Number of iterations taken with random descent direction: {i}")
print(f"Value of f at the minimum: {f_x}")
print(f"Value of x at the minimum: {x}")


######################################################################################################################

# TAKE HOME MESSAGE :-
# Gradient descent will always converge as long as descent direction is valid.


######################################################################################################################
# Varients of Gradient Descent FIRST ORDER METHODS 
######################################################################################################################



