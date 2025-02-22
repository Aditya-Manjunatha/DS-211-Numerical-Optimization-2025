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


# We will create a contour plotting function that does not save
def plot_contour_subplot(X, y, mse_lin_reg, id):

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

    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title(f'Contour Plot {id}')
    #plt.show()


# Make a batch selector function with batch size and batch index as parametrs
# The batch_selector must return X and y from batch_size[batch_index - 1 : batch_index]

def batch_selector(X, y, batch_size, batch_index):
    num_samples = X.shape[0]
    start = batch_index * batch_size
    end = start + batch_size 
    end = min(end, num_samples)
    return X[start:end, :], y[start:end]


# Make a 4*5 plot which shows the contour of the least squares objective function using one batch at a time
# Use the plot_contour_subplot function to plot the contours
"""
4*5 becuse we have 100 / 5 batches = 20 batches
So for each batch we have a plot for its contour
"""
def plot_contour_batches(X, y, mse_lin_reg):
    batch_size = 5
    plt.figure(figsize=(9, 9))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        X_batch, y_batch = batch_selector(X, y, batch_size, i)
        plot_contour_subplot(X_batch, y_batch, mse_lin_reg, i)
        plt.tight_layout()
    plt.savefig('contour_mse_4*5.png')
    plt.show()

#plot_contour_batches(X, y, mse_lin_reg)

"""
We can see that the contours are different for each batch EVEN THOUGH THE "THETA" IS SAME
This is because the data points in each batch are different
"""


theta_0 = np.array([[0], [0]]) 
step_size = 0.5
n_epochs = 5
batch_size = 5
n_batches = X.shape[0] // batch_size
n_steps = X.shape[0] // batch_size

# n_steps per epoch is n_samples // batch_size
# Because in each iteration we are seeing only one batch
# So in one epoch we have n_samples // batch_size iterations

"""
Essentially number of iterations inside each epoch is the same as number of batches
Which is why we have n_batches = n_steps and in line 118, we iterate over batches
"""


def batch_gradient_descent(X, y, theta_0, step_size, n_steps, mse_lin_reg, grad_mse_lin_reg, batch_size):
    theta = theta_0
    path = [theta]
    n_batches = X.shape[0] // batch_size

    for epoch in range(n_epochs):

        # Contains gradients for all batches in previous eopch
        # Ideally they all should be the same
        # But we saw in the previous 3*3 plot, that they are different
        dir_descent = np.zeros((theta.shape[0], n_batches))

        for batch_index in range(n_batches):
            X_batch, y_batch = batch_selector(X, y, batch_size, batch_index)

            """
            There was an issue in sirs code here, grad_mse was giving a vector of shape (2, 1), but we need a vector of shape (2, )
            """
            grad = grad_mse_lin_reg(X_batch, y_batch, theta)
            dir_descent[:, batch_index] = grad.flatten()

            # We dont ascend here. We just collect the descent directions for each batch
            # Later once dir_Descent in collected, we average them and then descend on that
            #theta = theta - step_size * grad_mse_lin_reg(X_batch, y_batch, theta)

            """
            But later sir told that it doesnt make a difference in datasets which are convex in nature
            So lets descent now only
            """
            theta = theta - step_size * grad
            path.append(theta)

        final_dir_descent = np.mean(dir_descent, axis=1)

        if np.linalg.norm(final_dir_descent) < 1e-6:
            break

        #theta = theta - step_size * final_dir_descent

        return theta, path
    

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
    plt.plot(path[:, 0], path[:, 1], 'kx', markersize=5, label='Path')

    # Plot final solution as a red circle
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, label='Final Solution')
    plt.legend()
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title(f'Batch_ID {id}')
    #plt.savefig('batch_SGD_path.png')
    #plt.show()

theta, path = batch_gradient_descent(X, y, theta_0, step_size, n_steps, mse_lin_reg, grad_mse_lin_reg, batch_size)

#print(theta)
#print(path)


def plot_contour_batches(X, y, mse_lin_reg, path):
    batch_size = 5
    plt.figure(figsize=(9, 9))
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        X_batch, y_batch = batch_selector(X, y, batch_size, i)
        plot_contour_subplot(X_batch, y_batch, mse_lin_reg, path, i)
    plt.tight_layout()
    plt.savefig('SBGD_subplots_with_path.png')
    plt.show()


plot_contour_batches(X, y, mse_lin_reg, path)


