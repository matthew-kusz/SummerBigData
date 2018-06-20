# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random
import time
import argparse

####### Global variables #######
'''
parser = argparse.ArgumentParser()
parser.add_argument('Lambda', help = 'Lambda / 10000000 (ie 1 = 1e-7), Adjust to prevent overfitting.', type = float)

args = parser.parse_args()
'''
global_step = 0
global_image_dim = 64
global_image_channels = 3
global_patch_dim = 8
global_num_patches = 50000
global_visible_size = 0
global_hidden_size = 36
global_lambda = 1e-5
global_num_classes = 4

# print 'You chose', args     

####### Definitions #######
# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Softmax
def hypo(value):
	# To prevent overflow subract the max number from each element in the array
	constant = value.max()
	return (np.exp(value - constant)/ np.reshape(np.sum(np.exp(value - constant), axis = 1), (m, 1)))

# Regularized cost function
def reg_cost(theta2, arr_x, arr_y):
	# Change our weights and bias values back into their original shape
	arr_W1, arr_b1 = reshape(theta2)

	# Find our hypothesis
	h = feedforward(arr_W1, arr_b1, arr_x)

	# Calculate the cost	
	cost1 = np.sum((-1.0 / m) * np.multiply(arr_y, np.log(h)))
	cost2 = (global_lambda / (2.0)) * (np.sum(np.multiply(arr_W1, arr_W1)))

	cost = cost1 + + cost2

	return cost

# Feedforward
def feedforward(W1, b1, arr_x):

	'''
	Tile function allows us to duplicate our rows to the proper dimensions without requiring a for loop.
	This enables each row in our dot product to receive the same bias term. If it were a (25, 10000) array it is equivalent to adding 
	our bias column to each dot product column with just + b1 (since b1 starts as a column).
	'''

	a3 = hypo(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))       # (m, 4) matrix

	return a3

# Backpropagation
def backprop(theta2, arr_x, arr_y):
	# Our backprop is using both softmax and logistic regression
	# To keep track of our iterations
	global global_step
	global_step += 1
	if (global_step % 50 == 0):
		print 'Global step: %g' %(global_step)

	# Change our weights and bias values back into their original shape
	arr_W1, arr_b1 = reshape(theta2)
	
	a3 = feedforward(arr_W1, arr_b1, arr_x)

	# Following previous method for computing the deltas
	arr_ones = np.ones((len(arr_x), 1))
	arr_x1 = np.hstack((arr_ones, arr_x))
	arr_W1b1 = np.hstack((arr_b1, arr_W1))

	# Compute the partial derivatives
	pd_W1 = np.dot((arr_y - a3).T, arr_x1)     # (4, m)

	del_W1 = (-1.0 / m) * pd_W1 + global_lambda * arr_W1b1

	# Changed the gradients into a one dimensional vector
	del_b1 = np.ravel(del_W1[:, : 1])
	del_W1 = np.ravel(del_W1[:, 1: ])

	D_vals = np.concatenate((del_W1, del_b1))
	return D_vals

# Set up our weights and bias terms
def weights_bias():
	'''
	Initialize parameters randomly based on layer sizes.
	We'll choose weights uniformly from the interval [-r, r]
	'''	
	r  = 0.12 
	random_weight1 = np.random.rand(global_num_classes, global_visible_size)     # (4, ?) matrix
	random_weight1 = random_weight1 * 2 * r - r

	# Set up our bias term
	bias1 = np.random.rand(global_num_classes, 1)    # (4, 1) matrix
	bias1 = bias1 * 2 * r - r

	# Combine these into a 1-dimension vector
	random_weight1_1D = np.ravel(random_weight1)
	bias1_1D = np.ravel(bias1)


	# Create a vector theta = W1 + b1
	theta_vals = np.concatenate((random_weight1_1D, bias1_1D))		
	
	return theta_vals

# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_num_classes * global_visible_size], (global_num_classes, global_visible_size))
	b1 = np.reshape(theta[global_num_classes * global_visible_size: global_num_classes * global_visible_size + global_num_classes],
		(global_num_classes, 1))

	
	return W1, b1

# Generate training data
def gen_train_data():
	data = scipy.io.loadmat('provided_data/stlTrainSubset.mat')
	labels = data['trainLabels']
	images = data['trainImages']
	num_images = int(data['numTrainImages'])

	# reformat our images array
	ord_img = np.zeros((num_images, 64, 64, 3))
	for i in range(num_images):
		ord_img[i] = images[:, :, :, i]

	print 'Dimensions of our training data: ', ord_img.shape

	return labels, ord_img, num_images

# Generate testing data
def gen_test_data():
	data = scipy.io.loadmat('provided_data/stlTestSubset.mat')
	labels = data['testLabels']
	images = data['testImages']
	num_images = int(data['numTestImages'])

	# reformat our images array
	ord_img = np.zeros((num_images, 64, 64, 3))
	for i in range(num_images):
		ord_img[i] = images[:, :, :, i]

	print 'Dimensions of our testing data: ', ord_img.shape

	return labels, ord_img, num_images

# Import the files we want and reshape them into the correct dimension
train = np.genfromtxt('outputs/convPoolTrainFeaturesSize2000StepSize50')
train = np.reshape(train, (400, 2000, 3, 3))
train = np.swapaxes(train, 0, 1)
train = np.reshape(train, (train.shape[0], len(train.ravel()) / train.shape[0]))
print 'Dimensions of train', train.shape

test = np.genfromtxt('outputs/convPoolTestFeaturesSize3200StepSize50')
test = np.reshape(test, (400, 3200, 3, 3))
test = np.swapaxes(test, 0, 1)
test = np.reshape(test, (test.shape[0], len(test.ravel()) / test.shape[0]))
print 'Dimensions of test', test.shape

global_visible_size = train.shape[1]

# Generate our training data
train_labels, __, __ = gen_train_data()
# Used to set up grad_check (works for full data set)
train_labels = train_labels[0: train.shape[0]]

# Generate our testing data
test_labels, __, __ = gen_test_data()
# Used to set up grad_check (works for full data set)
test_labels = test_labels[0: test.shape[0]]

# Set up the filename we want to use
# filename = 'outputs/finalWeightsL1e-4B3Size60000HL200SOFT.out'

# Need to know how many training inputs we have
m = train.shape[0]

# Create our weights and bias terms
theta = weights_bias()

# Set up an array that will be either 1 or 0 depending on what we are looking at
y_vals_train = np.zeros((len(train_labels), global_num_classes))
for i in range(global_num_classes):
	# Set up an array with the values that stand for each number
	arr_num = [1, 2, 3, 4]
	
	for j in range(len(train_labels)):
		if (train_labels[j] == arr_num[i]):
			y_vals_train[j, i:] = 1
		
		else:
			y_vals_train[j, i:] = 0

'''
# Check that our cost function is working
cost_test = reg_cost(theta, train, y_vals_train)
print cost_test
# We had a cost value of 1.41

# Gradient checking from scipy to see if our backprop function is working properly. Theta_vals needs to be a 1-D vector.
print scipy.optimize.check_grad(reg_cost, backprop, theta, train, y_vals_train)
# Recieved a value of 1.23e-6
'''

print 'Cost before minimization: %g' %(reg_cost(theta, train, y_vals_train))
time_start2 = time.time()

# Minimize the cost value
minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta, method = 'L-BFGS-B', tol = 1e-4, jac = backprop, args = (train, y_vals_train)) #options = {"disp":True}
theta_new = minimum.x

print 'Cost after minimization: %g' %(reg_cost(theta_new, train, y_vals_train))
time_finish2 = time.time()

# Save to a file to use later
# np.savetxt(filename, theta_new, delimiter = ',')

print 'Total time for minimization = %g seconds' %(time_finish2 - time_start2)
'''
'''
# Find the probabilities for each digit
# We need to reshape our theta values
final_W1, final_W2, final_b1, final_b2 = reshape(theta_new)

# Need to know how many testing inputs we have
m = len(test)
prob_all, nano = feedforward(final_W1, final_W2, final_b1, final_b2, test)
print prob_all.shape
# Find the largest value in each column
best_prob = np.zeros((len(prob_all), 1))
for i in range (len(prob_all)):
	best_prob[i, 0] = np.argmax(prob_all[i, :])
	 
# Find how accurate our program was
correct_guess = np.zeros((global_num_classes, 1))
for i in range(len(best_prob)):
	print best_prob[i] + 1, int(test_labels[i])
	if (best_prob[i] + 1 == int(test_labels[i])):
		correct_guess[int(test_labels[i]) - 1] = correct_guess[int(test_labels[i]) - 1] + 1

# Find how many of each image of our array test_labels has
y_digits = np.zeros((global_num_classes, 1))
for i in range(global_num_classes):
	for j in range(len(test_labels)):
		if (test_labels[j] == i + 1):
			y_digits[i] = y_digits[i] + 1
print y_digits
# Calculate the percentage
for i in range(len(correct_guess)):
	correct_guess[i] = (correct_guess[i] / y_digits[i]) * 100

# Check the results
print correct_guess

# Calculate the average accuracy
avg_acc = 0
for i in range(len(correct_guess)):
	avg_acc = avg_acc + correct_guess[i]

avg_acc = avg_acc / len(correct_guess)
print avg_acc
