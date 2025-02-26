import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return 4*x[0]**2 + x[1]**2 


# Plot the function
x = np.linspace(-4, 4, 50)
y = np.linspace(-4, 4, 50)
X, Y = np.meshgrid(x, y)
Z = f1([X, Y])
# fig = plt.figure()
# plt.contourf(X, Y, Z, levels = np.linspace(0, 10, 50), cmap='Blues')
# plt.colorbar()
# plt.savefig('f1.png')
# plt.show()

#############################################
# Task :- Implement coordinate descent algorithm
#############################################

def exact_line_search(f, x, d, alpha_inc = 1e-3):
    def g(alpha):
        return f(x + alpha * d)
    
    alpha = 0.0

    # Most simple algorithm

    while g(alpha) > g(alpha + alpha_inc):
        alpha += alpha_inc

    # If it was increasing when using positive alpha, then we need to go towards left of alpha = 0
    if alpha == 0:
        while g(alpha) > g(alpha - alpha_inc):
            alpha -= alpha_inc

    # Here alpha can be poistive or negative beacuse 'd' need not be a descent direction like in gradient descent
    # Which is why over there we always take positive alpha

    return alpha

def coordinate_descent(f, x0, tol=1e-6):
    x = x0
    n = len(x)
    path = [x]

    for i in range(n):
        
        # Define direction of search
        d = np.zeros(n)
        d[i] = 1

        # Get alpha which is the answer for the exact_line_search
        alpha = exact_line_search(f, x, d)

        x = x + alpha * d
        path.append(x)

    return x, path

# x0 = np.array([-1, -1])
# x, path = coordinate_descent(f1, x0)
#print(f"Minimum point for f1: {x}")
# fig = plt.figure()
# plt.contourf(X, Y, Z, levels = np.linspace(0, 10, 50), cmap='Blues')
# plt.colorbar()
# path = np.array(path)
# plt.plot(path[:, 0], path[:, 1], 'ro-')
# plt.savefig('f1_path.png')
# plt.show()


#############################################

def f2(x):
    return 4*x[0]**2 + x[1]**2 - 2*x[0]*x[1]


x, path = coordinate_descent(f2, np.array([-1, -1]))
print(f"Minimum point for f2: {x}")

fig = plt.figure()
Z = f2([X, Y])
plt.contourf(X, Y, Z, levels = np.linspace(-10, 10, 50), cmap='Blues')
plt.colorbar()
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], 'ro-')
plt.savefig('f2_path.png')
plt.show()


#############################################
# This method did not go to the minimum point !!!!

# This is beacuse for f2, there is a coupling between the variables

# f1 was a SEPARABLE function, i.e. the variables were not coupled

# f2 is a NON-SEPARABLE function, i.e. the variables are coupled
#############################################
