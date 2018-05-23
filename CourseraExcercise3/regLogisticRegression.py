# Import the necessary pachages
import scipy.io
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import sys, traceback, atexit

# Extract the provided data. We need to use scipy since the data is in a matlab file format
data = scipy.io.loadmat('ex3data1.mat')

# The x array is under the name 'X', the y array is under the name 'y'
x_vals = data['X']
y_vals = data['y']


####### Regularized logistic regression #######
# Sigmoid equation
def sigmoid(arr_theta, arr_x):
	return 1.0 / (1.0 + np.exp(-np.dot(arr_x, arr_theta.T)))

# Cost function
def cost(arr_theta, arr_x, arr_y, lambda1):
	arr_theta = np.reshape(arr_theta, (1, len(arr_theta)))
	h = sigmoid(arr_theta, arr_x)
	first_half = np.dot(arr_y.T, np.log(h))
	second_half = np.dot((1 - arr_y).T, np.log(1 - h))
	cost = (-1.0 / m) * (first_half + second_half) + (lambda1 / (2 * m)) * (np.dot(arr_theta, arr_theta.T) - arr_theta[0, 0] ** 2)
	return np.asscalar(cost)

# Cost function gradient
def cost_gradient(arr_theta, arr_x, arr_y, lambda1):
	arr_theta = np.reshape(arr_theta, (1, len(arr_theta)))
	h = sigmoid(arr_theta, arr_x)
	temp = (1.0 / m) * np.dot((h - arr_y).T, arr_x)	
	gradient = (1.0 / m) * np.add(np.dot((h - arr_y).T, arr_x), (lambda1 / m) * arr_theta)
	gradient[0, 0] = temp[0, 0]
	print gradient.flatten()
	return gradient.flatten()



####### One-vs-all Classification #######
lambda1 = 10.0
# Add a column of ones to our array of x_vals
m = len(x_vals)   # Number of training examples (rows)
arr_ones = np.ones((m, 1))
x_vals = np.hstack((arr_ones, x_vals))

print x_vals

# Set up our theta vector
n = len(x_vals[0])  # Number of columns
theta_vals = np.zeros((1, n))
'''
print x_vals.shape
print y_vals.shape
print theta_vals.shape
'''
y_vals_train = np.zeros((len(y_vals), 1))

for i in range(len(y_vals)):
	if (y_vals[i] == 10):
		y_vals_train[i] = 1
	
	else:
		y_vals_train[i] = 0

#test = cost_gradient(theta_vals, x_vals, y_vals_train, lambda1)
#print test
#test2 = cost(theta_vals, x_vals, y_vals_train, lambda1)
#print test2

minimum = scipy.optimize.minimize(fun = cost, x0 = theta_vals, method = 'BFGS', jac = cost_gradient, args = (x_vals, y_vals_train, lambda1), tol = 1e-4)#, options = {'disp': True}

#print minimum.x
