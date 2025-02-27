import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return 4*x[0]**2 + x[1]**2 

def f2(x):
    return 4*x[0]**2 + x[1]**2 - 2*x[0]*x[1]

def grad_f1(x):
    grad = np.array([8*x[0], 2*x[1]])
    return grad

def grad_f2(x):
    grad = np.array([8*x[0] - 2*x[1], 2*x[1] - 2*x[0]])
    return grad 

def hessian_f1(x):
    """
    Hessian of the function f1(x) = 4*x[0]**2 + x[1]**2

    Parameters:
    x : ndarray
        Input vector

    Returns:
    hessian : ndarray
        Hessian matrix
    """
    hessian = np.array([[8, 0], [0, 2]])
    return hessian

def hessian_f2(x):
    """
    Hessian of the function f2(x) = 4*x[0]**2 + x[1]**2 - 2*x[0]*x[1]

    Parameters:
    x : ndarray
        Input vector

    Returns:
    hessian : ndarray
        Hessian matrix
    """
    hessian = np.array([[8, -2], [-2, 2]])
    return hessian

#######
# Define a function for doing conjugate gradient 
#######
def conjugate_gradient(f, grad_f, hessain_f, x0, tol = 1e-6):

    r = grad_f(x0)
    p0 = -r 
    k = 0
    p = p0
    x = x0 
    path = [x0]

    while np.linalg.norm(r) > tol :

        alpha = - (r.T @ p) / ( p.T @ hessain_f(x) @ p )

        x = x + alpha * p 

        r = grad_f(x) 

        beta = (r.T @ hessain_f(x) @ p) / (p.T @ hessain_f(x) @ p)

        p = -r + beta *p

        path.append(x)

        k = k + 1

    return x, k, path


#####
# Visualize the path now :- Verify that it converges in 2 steps
#####

x, k, path = conjugate_gradient(f2, grad_f2, hessian_f2, np.array([-1, -1]))
print(f"Minimum point for f2: {x}")


x = np.linspace(-4, 4, 50)
y = np.linspace(-4, 4, 50)
X, Y = np.meshgrid(x, y)
fig = plt.figure()
Z = f2([X, Y])
plt.contourf(X, Y, Z, levels = np.linspace(-10, 10, 50), cmap='Blues')
plt.colorbar()
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], 'ro-')
plt.savefig('f2_conjugate_gradient.png')
plt.show()



#####
# Efficient Conjugate Gradient :- Algorithm 5.2 in text book 
####
def conjugate_gradient_efficient(f, grad_f, hessain_f, x0, tol = 1e-6):

    r = grad_f(x0)
    p0 = -r 
    k = 0
    p = p0
    x = x0 
    path = [x0]
    r_old = grad_f(x0)

    while np.linalg.norm(r_old) > tol :

        alpha = - (r.T @ r) / ( p.T @ hessain_f(x) @ p )

        x = x + alpha * p 

        r_new = r_old + alpha * hessain_f(x) @ p

        beta = (r_new.T @ r_new) / (r_old.T @ r_old)

        p = -r_new + beta *p

        path.append(x)

        k = k + 1

        r_old = r_new

    return x, k, path


### 37:00 :- Assgn details
### Cant be used if problem is not convex qaudratic problem

# Summary of course :- 45:00

# Be ready to derive an SR1 or SR2 update

# x, k, path = conjugate_gradient_efficient(f2, grad_f2, hessian_f2, np.array([-1, -1]))
# print(f"Minimum point for f2: {x}")


# x = np.linspace(-4, 4, 50)
# y = np.linspace(-4, 4, 50)
# X, Y = np.meshgrid(x, y)
# fig = plt.figure()
# Z = f2([X, Y])
# plt.contourf(X, Y, Z, levels = np.linspace(-10, 10, 50), cmap='Blues')
# plt.colorbar()
# path = np.array(path)
# plt.plot(path[:, 0], path[:, 1], 'ro-')
# plt.savefig('f2_conjugate_gradient_efficient.png')
# plt.show()