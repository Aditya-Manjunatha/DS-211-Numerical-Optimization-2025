import numpy as np
import matplotlib.pyplot as plt
from datamaker import RegressionDataMaker

# Create an instance of the RegressionDataMaker class
data_maker = RegressionDataMaker(n_samples=100, n_features=1, noise_level=0.1,seed = 42)

# Generate data
X, y, coefs = data_maker.generate_data()

X = data_maker.make_bias_column(X)

# Make a least squares objective function
def mse_lin_reg(X, y, theta):
    n_samples = X.shape[0]
    return np.sum((X @ theta - y) ** 2) / n_samples

# Make a gradient of the objective function
def grad_mse_lin_reg(X, y, theta):
    n_samples = X.shape[0]
    return 2 * X.T @ (X @ theta - y) / n_samples


def batch_selector(X, y, batch_size, batch_index):
    num_samples = X.shape[0]
    start = batch_index * batch_size
    end = start + batch_size 
    end = min(end, num_samples)
    return X[start:end, :], y[start:end]    

def plot_contour_subplot(X, y, mse_lin_reg, path, id):

    theta0_vals = np.linspace(-5, 5, 100)  
    theta1_vals = np.linspace(-5, 5, 100)
    theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)

    # Number of rows in j_vals should be number of theta1 vals = theta0_grid.shape[0] = theta1_grid.shape[1]
    # Number of columns in j_vals should be number of theta0 vals = theta0_grid.shape[1] = theta1_grid.shape[0]

    #print(theta0_grid.shape) # 100, 100
    #print(theta1_grid.shape)    # 100, 100

    j_vals = np.zeros((theta0_grid.shape[0], theta1_grid.shape[0]))
    for i in range(theta0_grid.shape[0]):
        for j in range(theta1_grid.shape[0]):
            # mse_lin_reg expects theta to be of shape (2, 1) as X is of shape (n_samples, 2)
            theta = np.array([[theta0_grid[i, j]], [theta1_grid[i, j]]])
            j_vals[i, j] = mse_lin_reg(X, y, theta)

    plt.contourf(theta0_grid, theta1_grid, j_vals, levels=np.arange(0, 100, 10), cmap='Blues')

    # Plot the path
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'kx', markersize=3, label='Path')

    # Plot final solution as a red circle
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=5, label='Final Solution')
    plt.legend()
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title(f'Batch_ID {id}')
    #plt.savefig('batch_SGD_path.png')
    #plt.show()


def plot_contour_batches(X, y, mse_lin_reg, path, beta_1, beta_2):
    batch_size = 5
    plt.figure(figsize=(12, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        X_batch, y_batch = batch_selector(X, y, batch_size, i)
        plot_contour_subplot(X_batch, y_batch, mse_lin_reg, path, i)
    plt.tight_layout()
    plt.savefig(f'Adam beta_1 = {beta_1}, beta_2 = {beta_2}, step_size = {step_size}.png')
    plt.show()


theta_0 = np.array([[2], [2]]) 
step_size = 0.001
n_epochs = 5
batch_size = 5

"""
So in both the methods we saw before SGD with and without momentum
There was no control on the step size, it was fixed

But if gradient is skewed in a particular direction, we obviosuly move more in the STEEPER direction

But very less in the other SHALLOWER direction

We need to dynamically adjust the step size for each direction

IF the gradient is skewed in a particular direction, we need to move LESS in that direction

This is what RMSProp does
"""

def Adam(X, y, theta_0, step_size, mse_lin_reg, grad_mse_lin_reg, batch_size, beta_1, beta_2):
    
    theta = theta_0
    path = [theta]
    beta_1 = beta_1
    beta_2 = beta_2
    n_batches = X.shape[0] // batch_size
    i = 0

    for epoch in range(n_epochs):

        avg_dir_descent = np.zeros((theta.shape[0], 1))
        avg_dir_descent_sq = np.zeros((theta.shape[0], 1))

        for batch_index in range(n_batches):

            X_batch, y_batch = batch_selector(X, y, batch_size, batch_index)

            dir_descent = grad_mse_lin_reg(X_batch, y_batch, theta)

            avg_dir_descent = beta_1*avg_dir_descent + (1 - beta_1)*dir_descent

            avg_dir_descent_sq = beta_2 * avg_dir_descent_sq + (1 - beta_2) * dir_descent**2

            avg_dir_descent = avg_dir_descent / (1 - beta_1**(i + 1))
            avg_dir_descent_sq = avg_dir_descent_sq / (1 - beta_2**(i + 1))

            adaptive_step_size = step_size / np.sqrt(avg_dir_descent_sq + 1e-8)

            theta = theta - adaptive_step_size * avg_dir_descent

            path.append(theta)
            i += 1

        if np.linalg.norm(avg_dir_descent) < 1e-6:
            break

        return theta, path
    



#plot_contour_batches(X, y, mse_lin_reg, path, beta_1 = beta_1, beta_2 = beta_2)

# Save the path
#np.save('Adam_path.npy', path)

def plot_rate_of_convergence(path, theta, id):
    diff = path - theta
    plt.plot(np.linalg.norm(diff, axis=1))
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(f'Rate of convergence {id}')
    plt.savefig(f'rate_of_convergence {id}.png')
    plt.show()

beta_1 = 0.9
beta_2 = 0.999
step_size = 0.001

theta, path = Adam(X, y, theta_0, step_size, mse_lin_reg, grad_mse_lin_reg,batch_size, beta_1 = beta_1, beta_2 = beta_2)

plot_rate_of_convergence(path, theta, id = f"beta_1 = {beta_1}, beta_2 = {beta_2}, step_size = {step_size}") 

# bet1, beta2 = 0 means no momentum and adam
# beta1 = 0 is rmsprop
# beta2 = 0 is SGD momentum
# beta1, beta2 = 0 is normal gradient descent
# beta1, beta2 = 0.9 is adam