# The oroblem statement is the to minimize 
# Obj function :- min x[0] + 2x[1]
# Subject to the constraints:
# x[0] <= 1
# x[0] + x[1] <= 2
# x[0], x[1] >= 0

# We are going to code the revised simplex method to solve the above problem
import numpy as np

A = np.array([[1, 0], [1, 1]])
b = np.array([1, 2]).reshape(-1, 1)
c = np.array([1, 2]).reshape(-1, 1)
A = np.hstack((A, np.eye(2)))
c = np.vstack((c, np.zeros((2, 1))))

m, n = A.shape

# Make a list of index of basic and non basic variables
# Intially we set the new vars we added as non basic and the original vars as basic
basic_vars = np.arange(0, n-m)
non_basic_vars = np.arange(n-m, n)

iter = 0
while True:
    print(f"Iteration : {iter}")
    B = A[:, basic_vars]
    N = A[:, non_basic_vars]
    x_B = np.linalg.inv(B) @ b
    c_B = c[basic_vars]
    c_N = c[non_basic_vars]

    if np.any(x_B < 0):
        print("Infeasible problem")
        exit()

    lambda_ = np.linalg.inv(B.T) @ c_B

    s_N = c_N - N.T @ lambda_

    if np.all(s_N >= 0):
        print("Optimal solution found")
        print(f"x_B : {x_B}")
        print(f"Basic variables : {basic_vars}")
        exit()

    # Need to find the index of the entering variable 
    entering_var_index = np.argmin(s_N)
    entering_var = non_basic_vars[entering_var_index]
    print(f"Entering variable : {entering_var}")


    # Need to select the leaving variable
    # For that need to solve Bd = A_q 
    A_q = A[:, entering_var].reshape(-1, 1) # Want Aq to be a column vector
    d = np.linalg.inv(B) @ A_q

    if np.all(d <= 0):
        print("Unbounded problem")
        exit()

    # Need to find the index of the leaving variable
    # Need to find the minimum ratio
    min_ratio = np.inf
    leaving_var_index = -1
    for i in range(m):
        if d[i] > 0:
            ratio = x_B[i] / d[i]
            if ratio < min_ratio:
                min_ratio = ratio
                leaving_var_index = i

    leaving_var = basic_vars[leaving_var_index]
    print(f"Leaving variable : {leaving_var}")

    basic_vars[leaving_var_index] = entering_var
    non_basic_vars[entering_var_index] = leaving_var

    basic_vars.sort()
    non_basic_vars.sort()

    print(f"Basic variables after iter: {basic_vars}")
    print(f"Non basic variables after iter: {non_basic_vars}")
    iter += 1
    print("\n")

"""
Optimal solution found
x_B : [[1.]
        [2.]]
Basic variables : [2 3]

So its saying that x0 and x1 are = 0 as basic vars are 2 and 3
x2 = 1 and x3 = 2
So the optimal solution is x0 = 0, x1 = 0, x2 = 1, x3 = 2
"""
