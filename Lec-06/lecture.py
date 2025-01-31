import numpy as np
import matplotlib.pyplot as plt
from datamaker import RegressionDataMaker

# Create an instance of the RegressionDataMaker class
data_maker = RegressionDataMaker(n_samples=100, n_features=1, noise_level=0.1,seed = 42)

# Generate data
X, y, coefs = data_maker.generate_data()
# Save X to a csv file
np.savetxt('X.csv', X, delimiter=',')
np.savetxt('y.csv', y, delimiter=',')

X = data_maker.make_bias_column(X)
np.savetxt('X_with_bias.csv', X, delimiter=',')

# Save the data to a csv file
#data_maker.save_data(X, y, 'data.csv')


"""
# Save the coefficients to a txt file
# Contains the coefs given by sklearn
# Will use it to compare with the coefs we get from our model
"""
data_maker.save_coefs(coefs, 'true_coefs.txt')

################################################################################
# Define the objective function
################################################################################

# Make a least squares objective function
def mse_lin_reg(X, y, theta):
    n_samples = X.shape[0]
    return np.sum((X @ theta - y) ** 2) / n_samples


# Confirm if :- 
# dim(X) = (n_samples, n_features + 1)  = (100, 2)
# dim(y) = (n_samples, 1) = (100, 1)
# dim(theta) = (n_features + 1, 1) = (2, 1)

# Make a gradient of the objective function
def grad_mse_lin_reg(X, y, theta):
    n_samples = X.shape[0]
    return 2 * X.T @ (X @ theta - y) / n_samples

# Make a gradient descent function
step_size = 0.01
n_steps = 100000

n_samples = X.shape[0]
n_features = X.shape[1]
theta_0 = np.array([[2], [2]])  # Random initialization


def gradient_descent(X, y, theta_0, step_size, n_steps, mse_lin_reg, grad_mse_lin_reg):

    theta = theta_0
    path = [theta]
    i = 0
    while np.linalg.norm(grad_mse_lin_reg(X, y, theta)) > 1e-6:

        theta = theta - step_size * grad_mse_lin_reg(X, y, theta)
        path.append(theta)

        if i > n_steps:
            break

        if i % 100 == 0:
            print(f"Step {i}, theta: {theta}, mse: {mse_lin_reg(X, y, theta)}")
        i += 1

    return theta, path


# Plot the contour of the least squares objective function

def plot_contour(X, y, path, mse_lin_reg, step_size):
    theta0_vals = np.linspace(-1, 3, 100)  # Adjusted bounds based on optimal (1,2)
    theta1_vals = np.linspace(0, 4, 100)
    theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)

    mse_vals = np.zeros_like(theta0_grid)
    for i in range(theta0_grid.shape[0]):
        for j in range(theta0_grid.shape[1]):
            theta = np.array([[theta0_grid[i, j]], [theta1_grid[i, j]]])
            mse_vals[i, j] = mse_lin_reg(X, y, theta)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(theta0_grid, theta1_grid, mse_vals, levels=np.linspace(mse_vals.min(), mse_vals.max(), 25), cmap='viridis')
    plt.colorbar(contour)

    # Convert path list to array for plotting
    path = np.array(path).squeeze()
    plt.plot(path[:, 0], path[:, 1], marker='x', color='black', linestyle='-', markersize=5, label="Gradient Descent Path")

    # Annotate iterations
    for i in range(0, len(path), max(1, len(path) // 10)):  # Show only 10 labels
        plt.text(path[i, 0], path[i, 1], str(i), fontsize=8, color='white')

    plt.scatter(path[-1, 0], path[-1, 1], color='red', s=100, label="Optimal Theta")  # Final point in red

    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title(f'Gradient Descent MSE Contour (Step Size = {step_size})')
    plt.legend()
    plt.savefig(f'gradient_descent_mse_contour_{step_size}.png')
    plt.show()

theta, path = gradient_descent(X, y, theta_0, step_size, n_steps, mse_lin_reg, grad_mse_lin_reg)

plot_contour(X, y, path, mse_lin_reg, step_size)



