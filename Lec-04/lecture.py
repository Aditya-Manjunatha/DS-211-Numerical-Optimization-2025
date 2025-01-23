import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2*x**2 + 3*x + 4

# Visualize this function on -10 to 10
x = np.linspace(-10, 10, 100)
y = f(x)

#plt.plot(x, y)
#plt.show()
#plt.savefig("visulaize_f(x).png")


# Define the derivative of the function
def df(x):
    return 4*x + 3

# We will be using an iterative approach to find the minima of the function :

# Make an initial guess of x_optim as x = 5
x_0 = 5

# print(f"f(x_0) = {f(x_0)}")
# print(f"gradient(x_0) = {df(x_0)}")

# The gradient is 23, which means, if we change x by 0.01, the function value will change by 23*0.01 = 0.23 in the direction in which we change x

# Define direction of descent 
dir_descent = ...

# Need to choose my step length :- It has to be positive for sure
step_length = ...

# Need to pick a step length length such that the function value decreases in x_new
# x_new = x_0 + step_length * dir_descent

# Expand the function around x_0
# f(x) = f(x_0) + df(x_0)*step_length*dir_descent + 1/2*d2f(x_0)^T * step_length**2 * dir_descent
#
#
# Let us just make a first order approximation of the function 

# f(x_new) = f(x_0) + df(x_0) * [step_length * dir_descent] + O(step_length*dir_descent**2)

# We dont know step_length, so lets assume this is a funciton of step_length

# f_x_new(step_length) = f(x_0) + df(x_0) * [step_length * dir_descent] + O(step_length*dir_descent**2)


# ################################################################################################################################################
# # GOAL : We want to find the step_length such that f_x_new(ste_length) is minimized

# # We know a lower bound on the step length, which is 0
# # But what abour upper bound ?
# # We need to put some conditions on the step length

# # We use Wolfe conditions to find the step length :-
# # So alpha must be such that there is sufficient decrease in the function value between x_0 and x_new

# ################################################################################################################################################
# # Consider f2(x1, x2) :

def f2(x): # x is a 2D vector
    return x[0]**2 + x[1]**2 + 0.5*x[0]*x[1]

# Visualize this function on -4 to 4
x1 = np.linspace(-4, 4, 100)
x2 = np.linspace(-4, 4, 100)

[x1, x2] = np.meshgrid(x1, x2)
y = f2(np.array([x1, x2]))

# To visualize the function, we can use a contour plot
# We will just use a sequential colormap becuase we just want to see whether it is a minimum or not 

# More colors your colot map has, more variations in functions you can see

plt.contourf(x1, x2, y, 200) # 200 is the number of contours
plt.colorbar()
plt.show()
plt.savefig("visulaize_f2(x1, x2).png")

################################################################################################################################################
# NEEDS TO BE FIXED :

# The issue is that many points are close to 0 in the map, and we cant see properly
# So we should clip the color map to some value

plt.contourf(x1, x2, y, 200) # 200 is the number of contours

cfp = plt.contourf(x1, x2, y, levels=np.linspace(0, 10, 10), cmap='Blues', extend='max', vmin=0, vmax=10)
cb = plt.colorbar(cfp)

plt.colorbar()
plt.show()
plt.savefig("visulaize_f2_clipped(x1, x2).png")
################################################################################################################################################

# Define the gradient of the function
def grad_f2(x):
    return np.array([2*x[0] + 0.5*x[1], 2*x[1] + 0.5*x[0]])


# Steps :-
# 1) Function to find the minimum of a 1D scalar function
# 2) Use that together with the gradient of the 2D scalar function to code the iterative gradient descent

# 1) Function to find the exact step length :- EXACT LINE SEARCH

# ALPHA_EXACT = argmin f(x_0 + alpha * dir_descent)

# Let us take some bounds on alpha
a = 0
b = 10

def golden_section_search(phi, a, b, tol=1e-6, max_iter=500):
    gr = (1 + 5**0.5) / 2
    
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    
    for i in range(max_iter):
        fc = phi(c)
        fd = phi(d)
        
        if fc < fd:
            b = d
            d = c
            c = b - (b - a) / gr
        else:
            a = c
            c = d
            d = a + (b - a) / gr
            
        # Check if interval is small enough
        if abs(b - a) < tol:
            break
    
    # Return the middle of the final interval
    x = (a + b) / 2

    return x

# 2) Use that together with the gradient of the 2D scalar function to code the iterative gradient descent

def iterative_gradient_descent(f, x_0, tol=1e-6, max_iter=100):
    x = x_0
    for i in range(max_iter):
        dir_descent = -grad_f2(x)
        step_length = golden_section_search(lambda alpha: f(x + alpha * dir_descent), 0, 10, tol=1e-6, max_iter=500)
        x = x + step_length * dir_descent
        if np.linalg.norm(grad_f2(x)) < tol:
            break
    return x


print(f"The minimum of the function is at {iterative_gradient_descent(f2, np.array([1, 1]), tol=1e-6, max_iter=500)}")
################################################################################################################################################
# Till now we did a first order gradient descent algorithm
# Now we will do a second order gradient descent algorithm

# f(x_new) = f(x_0) + df(x_0) * [step_length * dir_descent] + 1/2 * d2f(x_0) * [step_length * dir_descent]**2

# Check if this second order term above is correct or not

# Sir mentioned some quasi newton methods ???

# EXACT LINE SEARCH + BACKTRACKING WILL ALWAYS CONVERGE TO THE MINIMA FOR A CONVEX FUNCTION ONLY
################################################################################################################################################




