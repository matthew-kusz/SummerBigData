# THIS CODE DOES NOT HAVE A HIDDEN LAYER
# Import the necessary packages
import struct as st
import gzip
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random
import time

####### Global variables #######
global_num_images = 60000
global_step = 0
global_image_dim = 28
global_pooled_dim = 2
global_image_channels = 0
global_visible_size = 0    # Will be determined later
global_lambda = 1e-4
global_num_classes = 10 

####### Definitions #######
# Softmax
def hypo(value):
	# To prevent overflow subract the max number from each element in the array
	constant = value.max()
	return (np.exp(value - constant)/ np.reshape(np.sum(np.exp(value - constant), axis = 1), (m, 1)))

# Reading in MNIST data files
def read_idx(filename, n=None):
	with gzip.open(filename) as f:
		zero, dtype, dims = st.unpack('>HBB', f.read(4))
		shape = tuple(st.unpack('>I', f.read(4))[0] for d in range(dims))
		arr = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
		if not n is None:
			arr = arr[:n]
		return arr

# Regularized cost function
def reg_cost(theta2, arr_x, arr_y):
	# Change our weights and bias values back into their original shape
	arr_W1, arr_b1 = reshape(theta2)

	# Find our hypothesis
	h = feedforward(arr_W1, arr_b1, arr_x)

	# Calculate the cost	
	cost1 = np.sum((-1.0 / m) * np.multiply(arr_y, np.log(h)))
	cost2 = (global_lambda / (2.0)) * (np.sum(np.multiply(arr_W1, arr_W1)))

	cost = cost1 + cost2

	return cost

# Feedforward
def feedforward(W1, b1, arr_x):

	'''
	Tile function allows us to duplicate our rows to the proper dimensions without requiring a for loop.
	This enables each row in our dot product to receive the same bias term. If it were a (25, 10000) array it is equivalent to adding 
	our bias column to each dot product column with just + b1 (since b1 starts as a column).
	'''

	a3 = hypo(np.dot(arr_x, W1.T) + np.tile(np.ravel(b1), (m, 1)))       # (m, 10) matrix

	return a3

# Backpropagation
def backprop(theta2, arr_x, arr_y):
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
	pd_W1 = np.dot((arr_y - a3).T, arr_x1)     # (10, m)

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

	# Generate a seed so our random values remain the same through each run
	np.random.seed(7)

	random_weight1 = np.random.rand(global_num_classes, global_visible_size)     # (10, m) matrix
	random_weight1 = random_weight1 * 2 * r - r

	# Set up our bias term
	bias1 = np.random.rand(global_num_classes, 1)    # (10, 1) matrix
	bias1 = bias1 * 2 * r - r

	# Combine these into a 1-dimension vector
	random_weight1_1D = np.ravel(random_weight1)
	bias1_1D = np.ravel(bias1)


	# Create a vector theta_vals = W1 + b1
	theta_vals = np.concatenate((random_weight1_1D, bias1_1D))		
	
	return theta_vals

# Change our weights and bias terms back into their proper shapes
def reshape(theta):
	W1 = np.reshape(theta[0:global_num_classes * global_visible_size], (global_num_classes, global_visible_size))
	b1 = np.reshape(theta[global_num_classes * global_visible_size: global_num_classes * global_visible_size + global_num_classes],
		(global_num_classes, 1))

	return W1, b1

# Extract the MNIST training data sets
def get_train(size):
	x_vals = read_idx('provided_data/train-images-idx3-ubyte.gz', size)
	x_vals = x_vals / 255.0
	x_vals = np.reshape(x_vals, (x_vals.shape[0], (x_vals.shape[1] * x_vals.shape[2])))
	y_vals = read_idx('provided_data/train-labels-idx1-ubyte.gz', size)
	y_vals = np.reshape(y_vals, (len(y_vals), 1))
	print x_vals.shape, y_vals.shape
	
	return x_vals, y_vals

# Extract the MNIST testing data sets
def get_test(size):
	x_vals = read_idx('provided_data/t10k-images-idx3-ubyte.gz', size)
	x_vals = x_vals / 255.0
	x_vals = np.reshape(x_vals, (x_vals.shape[0], (x_vals.shape[1] * x_vals.shape[2])))
	y_vals = read_idx('provided_data/t10k-labels-idx1-ubyte.gz', size)
	y_vals = np.reshape(y_vals, (len(y_vals), 1))
	print x_vals.shape, y_vals.shape
	
	return x_vals, y_vals

# Confusion matrix
def ConfuMat(a3, y, amount):
	# Actual x Predicted
	a3avg = np.zeros((a3.shape[1], a3.shape[1]))
	# number in each feature
	numFeatures = np.zeros((a3.shape[1], 1))

	for i in range(m):
		a3avg[ y[i] ] += a3[i]
		numFeatures [ y[i] , 0 ] += 1
	
	a3avg /= numFeatures
	a3avg *= amount
	a3avg = np.rint(a3avg)
	
	yAxLabels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
	xAxLabels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


	fig, ax = plt.subplots()
	im = ax.imshow(a3avg, cmap="coolwarm", interpolation = 'none')

	# We want to show all ticks...
	ax.set_xticks(np.arange(len(xAxLabels)))
	ax.set_yticks(np.arange(len(yAxLabels)))
	# ... and label them with the respective list entries
	ax.set_xticklabels(xAxLabels)
	ax.set_yticklabels(yAxLabels)
	# Add x and y axis labels
	ax.set_xlabel('Predicted Object')
	ax.set_ylabel('Actual Object')
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(yAxLabels)):
	    for j in range(len(xAxLabels)):
		centerNumb = np.around(a3avg[i, j], decimals=4)
		text = ax.text(j, i, centerNumb, 
		               ha="center", va="center", color="w")

	ax.set_title("Confusion Matrix for Convolutional Neural Network")
	fig.tight_layout()
	plt.savefig('images/ConfusionMatrix.png', format='png')
	plt.show()

####### Code #######
# Import the files we want and reshape them into the correct dimension
train = np.genfromtxt('outputs/convPoolTrainFeaturesSize60000StepSize50.out')
train = np.reshape(train, (100, 60000, global_pooled_dim, global_pooled_dim))
train = np.swapaxes(train, 0, 1)
train = np.reshape(train, (train.shape[0], len(train.ravel()) / train.shape[0]))   # (60k, 4900)
print 'Dimensions of train', train.shape

test = np.genfromtxt('outputs/convPoolTestFeaturesSize10000StepSize50.out')
test = np.reshape(test, (100, 10000, global_pooled_dim, global_pooled_dim))
test = np.swapaxes(test, 0, 1)
test = np.reshape(test, (test.shape[0], len(test.ravel()) / test.shape[0]))   # (10k, 4090)
print 'Dimensions of test', test.shape

global_visible_size = train.shape[1]

# Extract the MNIST training data sets
train_images, train_labels = get_train(global_num_images)
num_train_images = len(train_images)

# Reshape our images so that they are (num_train_images , 28, 28)
train_images = np.reshape(train_images, (len(train_images), len(train_images[0]) / global_image_dim, len(train_images[0]) / global_image_dim))

# Used to set up grad_check (works for full data set)
train_labels = train_labels[0: train.shape[0]]

# Extract the MNIST testing data sets
test_images, test_labels = get_test(global_num_images)
num_test_images = len(test_images)

# Reshape our images so that they are (num_test_images , 28, 28)
test_images = np.reshape(test_images, (len(test_images), len(test_images[0]) / global_image_dim, len(test_images[0]) / global_image_dim))

# Used to set up grad_check (works for full data set)
test_labels = test_labels[0: test.shape[0]]

# Need to know how many training inputs we have
m = train.shape[0]

# Create our weights and bias terms
theta = weights_bias()

# Set up an array that will be either 1 or 0 depending on what we are looking at
y_vals_train = np.zeros((len(train_labels), global_num_classes))
for i in range(global_num_classes):
	# Set up an array with the values that stand for each label
	arr_num = [0, 1, 2, 3, 4, 5, 6, 7, 8 ,9]
	
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

print 'Total time for minimization = %g seconds' %(time_finish2 - time_start2)

# Find the probabilities for each image
# We need to reshape our theta values
final_W1, final_b1 = reshape(theta_new)

# Need to know how many testing inputs we have
m = len(test)
prob_all = feedforward(final_W1, final_b1, test)

# Find the largest value in each column
best_prob = np.zeros((len(prob_all), 1))
for i in range (len(prob_all)):
	best_prob[i, 0] = np.argmax(prob_all[i, :])
	 
# Find how accurate our program was
correct_guess = np.zeros((global_num_classes, 1))
for i in range(len(best_prob)):
	if (best_prob[i] == int(test_labels[i])):
		correct_guess[int(test_labels[i])] = correct_guess[int(test_labels[i])] + 1

# Find how many of each image our array test_labels has
y_digits = np.zeros((global_num_classes, 1))
for i in range(global_num_classes):
	for j in range(len(test_labels)):
		if (test_labels[j] == i):
			y_digits[i] = y_digits[i] + 1

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

# Let's create a confusion matrix
ConfuMat(prob_all, test_labels, y_digits)


