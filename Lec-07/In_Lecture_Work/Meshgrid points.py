import numpy as np

theta0_vals = np.linspace(-5, 5, 10)  
theta1_vals = np.linspace(-5, 5, 20)
theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)

print(theta0_grid.shape)
print(theta1_grid.shape)

"""
1) Number of rows in theta0_grid = # of points in theta1_vals
2) Number of columns in theta0_grid = # of points in theta0_vals

3) Number of rows in theta1_grid = # of points in theta1_vals
4) Number of columns in theta1_grid = # of points in theta0_vals

So think of this grid like the X and Y axis of a graph.
On X axis i have 10 points
On Y axis i have 20 points

So in the grid :- 
1) I create 20 copies of the X axis for each Y value
2) I create 10 copies of the Y axis for each X value

So the grid is 20 x 10

theta0_grid = 20 x 10 :- 10 X points copied 20 times
theta1_grid = 20 x 10 :- 20 Y points copied 10 times
"""


"""
x_grid.shape[0] and y_grid.shape[0] both give the number of rows (which corresponds to y values)

x_grid.shape[1] and y_grid.shape[1] both give the number of columns (which corresponds to x values).
"""

# Take X from the csv file
X = np.loadtxt('/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/6th Semester/Numerical Optimization/DS-211-Numerical-Optimization-2025/X_with_bias.csv', delimiter=',')
y = np.loadtxt('/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/6th Semester/Numerical Optimization/DS-211-Numerical-Optimization-2025/y.csv', delimiter=',')

def mse_lin_reg(X, y, theta):
    n_samples = X.shape[0]
    return np.sum((X @ theta - y) ** 2) / n_samples


j_vals = np.zeros((theta0_grid.shape[0], theta1_grid.shape[0]))

print(j_vals.shape)
# for i in range(theta1_grid.shape[0]): # iterate over the rows of the grid
#     for j in range(theta0_grid.shape[1]): # iterate over the columns of the grid
#         theta = np.array([[theta0_grid[i, j]], [theta1_grid[i, j]]])
#         j_vals[i, j] = mse_lin_reg(X, y, theta)
#         print(f"theta0: {theta0_grid[i, j]}, theta1: {theta1_grid[i, j]}, mse: {j_vals[i, j]}")
#         break
#     break

