# Define a class for creating data for the regression problem
# Use sklearn's make_regression function to generate data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class RegressionDataMaker:
    def __init__(self, n_samples = 100, n_features = 1, noise_level = 0.1, seed = 0, true_coefs = np.array([[1], [2]])):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_level = noise_level
        self.seed = seed
        self.true_coefs = true_coefs

    def generate_data(self):

        # What are we doing here, with b
        X, y, coefs = make_regression(n_samples=self.n_samples, n_features=self.n_features, noise=self.noise_level, random_state=self.seed, coef=True)
        coefs = self.true_coefs
        y = X @ coefs[1:] + coefs[0]
        return X, y, coefs

    # Save the generated data to a csv file
    def save_data(self, X, y, filename):
        np.savetxt(filename, np.column_stack((X, y)), delimiter=',')
        print(f"Data saved to {filename}")

    # Sve the coefficients to a txt file
    def save_coefs(self, coefs, filename):
        np.savetxt(filename, coefs, delimiter=',')
        print(f"Coefficients saved to {filename}")

    # Make a function to create a column of 1's for the bias in X
    def make_bias_column(self, X):
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        return X
    
    # Function to plot the generated data
    # Will only word if X has 1 feature
    def plot_data(self, X, y):
        plt.scatter(X, y, color='blue')
        plt.title('Regression Data')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()

    
