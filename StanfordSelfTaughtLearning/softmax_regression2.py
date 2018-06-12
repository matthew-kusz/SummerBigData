# Import the necessary packages
import struct as st
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random
import math
import grab_data
import time

####### Global variables #######
global_step = 0
global_input_size  = 28 * 28
global_hidden_size = 200
global_lambda = 1e-4;       # weight decay parameter         

####### Definitions #######
# Sigmoid function
def sigmoid(value):
	# To prevent overflow subract the max number from each element in the array
	constant = value.max()
	return (np.exp(value - constant)/ np.reshape(np.sum(np.exp(value - constant), axis = 1), (m, 1)))

# Regularized cost function
def reg_cost(thetaW2, arr_x, arr_y, thetaW1):

	# Change our weights and bias values back into their original shape
	arr_W1, arr_W2, arr_b1, arr_b2 = reshape(thetaW1, thetaW2)

	# Find our hypothesis
	h, a2 = feedforward(arr_W1, arr_W2, arr_b1, arr_b2, arr_x)

	# Calculate the cost	
	cost1 = np.sum((-1.0 / m) * np.multiply(arr_y, np.log(h)))
	cost3 = (global_lambda / (2.0)) * (np.sum(np.multiply(arr_W2, arr_W2)))
	cost = cost1 + cost3

	return cost

# Feedforward
def feedforward(W1, W2, b1, b2, arr_x):

	'''
	We will be running our sigmoid function twice.
	Tile function allows us to duplicate our rows to the proper dimensions without requiring a for loop.
	This enables each row in our dot product to receive the same bias term. If it were a (25, 10000) array it is equivalent to adding 
	our bias column to each dot product column with just + b1 (since b1 starts as a column).
	'''
	a2 = sigmoid(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))    # (m, 200) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, W2.T) + np.tile(np.ravel(b2), (m, 1)))       # (m, 5) matrix

	return a3, a2

# Backpropagation
def backprop(thetaW2, arr_x, arr_y, thetaW1):
	# To keep track of our iterations
	global global_step
	global_step += 1
	if (global_step % 50 == 0):
		np.savetxt(filename, theta, delimiter = ',')
		print 'Global step: %g' %(global_step)

	# Change our weights and bias values back into their original shape
	arr_W1, arr_W2, arr_b1, arr_b2 = reshape(thetaW1, thetaW2)
	
	a3, a2 = feedforward(arr_W1, arr_W2, arr_b1, arr_b2, arr_x)

	# Following the exact method given to us in the instructions
	arr_ones = np.ones((len(a2), 1))
	a2_1 = np.hstack((arr_ones, a2))
	arr_W2b2 = np.hstack((arr_b2, arr_W2))

	# Compute the partial derivatives
	pd_W2 = np.dot((arr_y - a3).T, a2_1)   # (5, 201)

	del_W2 = (-1.0 / m) * pd_W2 + global_lambda * arr_W2b2

	# Changed the gradients into a one dimensional vector
	del_b2 = np.ravel(del_W2[:, : 1])
	del_W21 = np.ravel(del_W2[:, 1: ])

	D_vals = np.concatenate((del_W21, del_b2))
	return D_vals

# Set up our weights and bias terms
def weights_bias():
	'''
	Initialize parameters randomly based on layer sizes.
	W1 and b1 will be taking from our autoencoder portion of this exercise
	We'll choose weights uniformly from the interval [-r, r]
	'''	
	r  = 0.12
	# math.sqrt(6) / math.sqrt(global_hidden_size + global_visible_size + 1);
	random_weight2 = np.random.rand(5, global_hidden_size)     # (5, 200) matrix      
	random_weight2 = random_weight2 * 2 * r - r

	# Set up our bias term
	bias2 = np.random.rand(5, 1)    # (5, 1) matrix
	bias2 = bias2 * 2 * r - r

	# Combine these into a 1-dimension vector
	random_weight2_1D = np.ravel(random_weight2)
	bias2_1D = np.ravel(bias2)

	thetas = np.genfromtxt('outputs/finalWeightsL3e-3B3Rho0.1Size60000HL200.out')
	random_weight1_1D = thetas[0:global_hidden_size * global_input_size]
	bias1_1D = thetas[2 * global_hidden_size * global_input_size: 2 * global_hidden_size * global_input_size + global_hidden_size]
	# Create a vector theta = W1 + W2 + b1 + b2
	theta_vals1 = np.concatenate((random_weight1_1D, bias1_1D))
	theta_vals2 = np.concatenate((random_weight2_1D, bias2_1D))	
	
	return theta_vals1 , theta_vals2

# Change our weights and bias terms back into their proper shapes
def reshape(theta1, theta2):
	W1 = np.reshape(theta1[0:global_hidden_size * global_input_size], (global_hidden_size, global_input_size))
	b1 =np.reshape(theta1[global_hidden_size * global_input_size: len(theta1)], (global_hidden_size, 1))
	W2 = np.reshape(theta2[0:global_hidden_size * 5], (5, global_hidden_size))
	b2 =np.reshape(theta2[global_hidden_size * 5: len(theta2)], (5, 1))
	return W1, W2, b1, b2

time_start = time.time()
# Import the file we want
data , labels_data = grab_data.get_data(60000, '04')
time_finish = time.time()
print 'Total time for obtaining data = %g' %(time_finish - time_start)

# We want to divide our data into halves, one for training and one for testing
data_points = len(data) / 2
train = data[0:data_points] 
labels_train = labels_data[0:data_points]
test = data[data_points:]
labels_test = labels_data[data_points:]

# Set up the filename we want to use
filename = 'outputs/finalWeightsL1e-4B3Size60000HL200SOFT.out'

# Need to know how many inputs we have
m = len(train)

# Create our weights and bias terms
theta1, theta2 = weights_bias()

# Set up an array that will be either 1 or 0 depending on which number we are looking at
y_vals_train = np.zeros((len(labels_train), 5))
for i in range(5):
	# Set up an array with the values that stand for each number
	arr_num = [0, 1, 2, 3, 4]
	
	for j in range(len(labels_train)):
		if (labels_train[j] == arr_num[i]):
			y_vals_train[j, i:] = 1
		
		else:
			y_vals_train[j, i:] = 0

'''
# Check that our cost function is working
cost_test = reg_cost(theta2, train, y_vals_train, theta1)
print cost_test
# We had a cost value of 5.6

# Gradient checking from scipy to see if our backprop function is working properly. Theta_vals needs to be a 1-D vector.
print scipy.optimize.check_grad(reg_cost, backprop, theta2, train, y_vals_train, theta1)
# Recieved a value of 1.45e-5
'''

print 'Cost before minimization: %g' %(reg_cost(theta2, train, y_vals_train, theta1))
time_start2 = time.time()
'''
# Minimize the cost value
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta2, method = 'L-BFGS-B', tol = 1e-4, jac = backprop, args = (train, y_vals_train, theta1)) #options = {"disp":True}
print minimum
theta_new = minimum.x

print 'Cost after minimization: %g' %(reg_cost(theta_new, train, y_vals_train, theta1))
time_finish2 = time.time()

# Save to a file to use later
# np.savetxt(filename, theta_new, delimiter = ',')

print 'Total time for minimization = %g' %(time_finish2 - time_start2)
'''
