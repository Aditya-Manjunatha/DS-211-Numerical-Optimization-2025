import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/adityamanjunatha/Library/CloudStorage/OneDrive-IndianInstituteofScience/IISc Semester/6th Semester/Numerical Optimization/DS-211-Numerical-Optimization-2025/Lec-02/real_estate_dataset.csv')

n_samples, n_features = df.shape 

columns = df.columns 

# Save the columns to a text file

np.savetxt('column_names.txt', columns, fmt = '%s')

# The columns are :-
"""
ID
Square_Feet
Num_Bedrooms
Num_Bathrooms
Num_Floors
Year_Built
Has_Garden
Has_Pool
Garage_Size
Location_Score
Distance_to_Center
Price

We only need Square_feet, Garage_size, Location_score, Distance_to_centre. Target is price
"""

X = df[["Square_Feet", 'Garage_Size', 'Location_Score', 'Distance_to_Center']]

# y is a numpy aray now
y = df['Price'].values 

# print(y)


# Build a linear model to predict price using X
n_samples, n_features = X.shape

# Make an array of coeffs of the size of n_features + 1 and initialize to 1
coefs = np.ones(n_features + 1)
# n + 1 to account(absorb) the bias

# Predict the price for each sample X
predictions_by_defn = X @ coefs[1:] + coefs[0]

# Append a column of 1's in X (similar to what chiru did) Predictions must be same
"""
X_before = [
    [feature1_1, feature2_1, feature3_1],
    [feature1_2, feature2_2, feature3_2],
    [feature1_3, feature2_3, feature3_3],
    ...
]
"""

X = np.hstack((np.ones((n_samples, 1)), X))

"""
X_after = [
    [1, feature1_1, feature2_1, feature3_1],
    [1, feature1_2, feature2_2, feature3_2],
    [1, feature1_3, feature2_3, feature3_3],
    ...
]
This is done to handle the bias term (intercept) in linear regression. Instead of having to separately add the bias term b in the equation:
Before :-
y = X @ coefs + b

After :-
y = X @ coefs

y = b + w1*x1 + w2*x2 + w3*x3

Now, 

y = w0*1 + w1*x1 + w2*x2 + w3*x3 

Where w0 is the bias = b

This allows us to handle all coefficients (including bias) in a single matrix multiplication operation: X @ coefs

Without haveing to write predictions = X @ coefs + b
Cutely we can write predictions = X @ coefs

"""

predictions = X @ coefs 

# Confirm if both are same
is_same = np.allclose(predictions_by_defn, predictions)

# Calculate the errors
errors = y - predictions

print(f"L2 norm of errors {np.linalg.norm(errors)}")

# But error is very high, how do i know if model is good or not ?
# Use relative error

relative_errors = errors / y 

# Turns out this also is high. It should ideally be close to 1
print(f"L2 norm of realtive_errors {np.linalg.norm(relative_errors)}")


# Find the MSE btw predictions and y
mse = np.mean((predictions - y)**2)
#print(mse) = 353620807434.83856



####################################################################################################################################################################################
                                                        # OPTIMIZATION PROBLEM :- FIND COEFS THAT MINIMIZE THE MSE
                                                        # THIS IS THE LEAST - SQUARES - PROBLEM

# Objective function :- the MSE formula


# What is a solution :-
# A set of coefficients in the space of n_features + 1 that minimize the objective function
# At the solution point, the gradient of the object wrt ... will be 0 AND
# Hessian will be PSD at the solution point

# How to find solution ?
# by searching for coeffs that make grad = 0
# OR set the gradient to 0, and solve the equation

loss_matrix = (y - X @ coefs).T @ (y - X @ coefs) / n_samples

# Take derivative of loss_matrix with coefs :- Prove this on paper
grad_matrix = -2/n_samples * X.T @ (y - X @ coefs)

# Set grad_matrix = 0 we get the NORMAL equation
# X.T @ X @ coefs - X.T @ y

coefs = np.linalg.inv(X.T @ X) @ X.T @ y 

# Save the coefs in a csv file
np.savetxt('coefs.txt', coefs)

# Make predictions using coefs
# Find error, relative error and their L2 norms
predictions =  X @ coefs

errors = y - predictions

rel_errors = errors / y 

print(f"L2 norm of errors {np.linalg.norm(errors)}")
print(f"L2 norm of realtive_errors {np.linalg.norm(rel_errors)}")

# Error is still not < 1. Then some data preprocessing, feature scaling, importance is needed to be done
# NOTICE that the coefficients are very large. So model is not good because of the data. Hand over to Data Scientists


##################################################################################################################################

# Use all the features in the dataset to build the linear model

X = df.drop(columns = ['Price'])
y = df['Price'].values

# Repeat on your own :-
# CODE HERE 
# Dont forget to append one's
X = np.hstack((np.ones((n_samples, 1)), X))
n_samples, n_features = X.shape

coefs = np.linalg.inv(X.T @ X) @ X.T @ y

loss_matrix = (y - X @ coefs).T @ (y - X @ coefs) / n_samples

grad_matrix = -2 / n_samples * X.T @ (y - X @ coefs)

# Set grad_matrix = 0 we get the NORMAL equation
# X.T @ X @ coefs - X.T @ y

# Save the coefs in a csv file
np.savetxt('coefs_all_cols.txt', coefs)

# Make predictions using coefs
# Find error, relative error and their L2 norms
predictions =  X @ coefs

errors = y - predictions

rel_errors = errors / y 

print(f"L2 norm of errors {np.linalg.norm(errors)}")
print(f"L2 norm of realtive_errors {np.linalg.norm(rel_errors)}")


#####################################################################################################################################

# The most expensive step is finding the inverse of X.T @ X
# Most of the time it is invertible. But if we have many dataponits and too few features. It is not invertible

# So we have two options to find the inverse :-

# Solve normal equation using matrix decomposition

# 1) 
# QR factorization
# X = QR
"""
Q, R = np.linalg.qr(X)

# Q will be orthogonal (Q.T @ Q = I) and R is upper triangular

# X.T @ X = R^T * Q^T * Q * R = R^T * R
# X.T @ y = R^T * Q^T * y 

# R * coefs = Q.T @ y
# This equaiton is of the form R * coefs = b
# Can be solved using Back - Substitution to find coefs

b = Q.T @ y 

# Print shape of b and R here
print(f"Shape of b {b.shape}")
print(f"Shape of R {R.shape}")

coefs_qr_loop = np.zeros(n_features + 1)

# Using Back Substitution method
# Assuming n_features includes the bias term, so it should be equal to R.shape[0] - 1
for i in range(n_features - 1, -1, -1):
    # Handle the last row separately to avoid empty slice
    if i == n_features - 1:
        coefs_qr_loop[i] = b[i] / R[i, i]
    else:
        coefs_qr_loop[i] = (b[i] - np.dot(R[i, i+1:], coefs_qr_loop[i+1:])) / R[i, i]

# Save the coefs_qr_loop into csv
np.savetxt('coefs_qr_loop.txt', coefs_qr_loop)

# Print L2 norms for the errors using these coefficients
predictions_qr = X @ coefs_qr_loop
errors_qr = y - predictions_qr
rel_errors_qr = errors_qr / y

print(f"L2 norm of errors (QR): {np.linalg.norm(errors_qr)}")
print(f"L2 norm of relative errors (QR): {np.linalg.norm(rel_errors_qr)}")
"""


# 2) Using SVD :- 

# X = U S V^T

# WANT X*coefs = y
# But X is not invertible, so need to do this X^T X business

# normal equation :- A*coeffs = X^T @ y
# Xdagger = A^-1 @ X^T

# A = X.T @ X is square, do its eigen decomp
# Turns out SVD OF X == EIGEN decomp of A

# TODO :- Calculate the coefs using SVD
# SVD of X
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# Compute the pseudo-inverse of the singular values
S_inv = np.diag(1 / S)

# Calculate the coefficients using SVD
# coefs = Pseudo inverse of X @ y
# Since X is not invertible, we use the pseudo inverse
coefs_svd = Vt.T @ S_inv @ U.T @ y

# Save the coefs_svd into a csv file
np.savetxt('coefs_svd.csv', coefs_svd)

# Print L2 norms for the errors using these coefficients
predictions_svd = X @ coefs_svd
errors_svd = y - predictions_svd
rel_errors_svd = errors_svd / y

print(f"L2 norm of errors (SVD): {np.linalg.norm(errors_svd)}")
print(f"L2 norm of relative errors (SVD): {np.linalg.norm(rel_errors_svd)}")


# WHAT DOES SOLVING Ax = b in the least squares sense mean ?

# Now that we have the coefs of the SVD method. Plot the predictions and the fit line for these coefficients
# First let us plot between square feet and price

# Create a range of square feet values for plotting
sq_ft_min = X[:, 1].min()
sq_ft_max = X[:, 1].max()
sq_ft_range = np.linspace(sq_ft_min, sq_ft_max, 100)

# Create prediction data where we only vary square feet
# Keep other features at their mean values
X_plot = np.zeros((100, X.shape[1]))
X_plot[:, 0] = 1  # bias term
X_plot[:, 1] = sq_ft_range  # square feet
# Set other features to their mean values
for i in range(2, X.shape[1]):
    X_plot[:, i] = np.mean(X[:, i])

# Generate predictions
predictions_plot = X_plot @ coefs_svd

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 1], y, alpha=0.5, label='Actual Data')
plt.plot(sq_ft_range, predictions_plot, color='red', linewidth=2, label='Linear Fit')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('House Price vs Square Feet')
plt.legend()
plt.show()
plt.savefig('square_feet_price.png')