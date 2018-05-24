import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Extract the provided data. We need to use scipy since the data is in a matlab file format
data = scipy.io.loadmat('ex3data1.mat')
data_thetas = scipy.io.loadmat('ex3weights.m')

'''
The x array is under the name 'X', the y array is under the name 'y'
x_vals ia a (5000, 400) array and y_vals is a (5000, 1) array
'''
x_vals = data['X']
y_vals = data['y']

'''
The array for theta1 is under the name 'Theta1' and the array for theta2 is under the nam 'Theta2'
theta1_vals is a (25, 401) array and theta2_vals is a (10, 26) array
'''
theta1_vals = data_thetas['Theta1']
theta2_vals = data_thetas['Theta2']

# Sigmoid equation
def sigmoid(arr_theta, arr_x):
	return 1.0 / (1.0 + np.exp(-np.dot(arr_x, arr_theta.T)))

# Add a column of ones to our array of x_vals
m = len(x_vals)    # Number of training examples (rows)
arr_ones = np.ones((m, 1))
x_vals = np.hstack((arr_ones, x_vals))

print x_vals.shape


