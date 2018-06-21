# THIS CODE DOES NOT HAVE A HIDDEN LAYER
# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.io
import random
import time

####### Global variables #######

global_step = 0
global_image_dim = 64
global_image_channels = 3
global_visible_size = 0    # Will be determined later
global_lambda = 1e-5
global_num_classes = 4   

####### Definitions #######
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

	cost = cost1 + cost2

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

	# Generate a seed so our random values remain the same through each run
	np.random.seed(7)

	random_weight1 = np.random.rand(global_num_classes, global_visible_size)     # (4, m) matrix
	random_weight1 = random_weight1 * 2 * r - r

	# Set up our bias term
	bias1 = np.random.rand(global_num_classes, 1)    # (4, 1) matrix
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

# Generate training data
def gen_train_data():
	data = scipy.io.loadmat('provided_data/stlTrainSubset.mat')
	labels = data['trainLabels']
	images = data['trainImages']
	num_images = int(data['numTrainImages'])

	# reformat our images array
	ord_img = np.zeros((num_images, global_image_dim, global_image_dim, global_image_channels))
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
	ord_img = np.zeros((num_images, global_image_dim, global_image_dim, global_image_channels))
	for i in range(num_images):
		ord_img[i] = images[:, :, :, i]

	print 'Dimensions of our testing data: ', ord_img.shape

	return labels, ord_img, num_images

# Import the files we want and reshape them into the correct dimension
train = np.genfromtxt('outputs/convPoolTrainFeaturesSize2000StepSize50')
train = np.reshape(train, (400, 2000, global_image_channels, global_image_channels))
train = np.swapaxes(train, 0, 1)
train = np.reshape(train, (train.shape[0], len(train.ravel()) / train.shape[0]))   # (m, 3600)
print 'Dimensions of train', train.shape

test = np.genfromtxt('outputs/convPoolTestFeaturesSize3200StepSize50')
test = np.reshape(test, (400, 3200, global_image_channels, global_image_channels))
test = np.swapaxes(test, 0, 1)
test = np.reshape(test, (test.shape[0], len(test.ravel()) / test.shape[0]))   # (m, 3600)
print 'Dimensions of test', test.shape

global_visible_size = train.shape[1]

# Generate our training data
train_labels, __, __ = gen_train_data()
# Used to set up grad_check (works for full data set)
train_labels = train_labels[0: train.shape[0]]

# Generate our testing data
test_labels, test_image, __ = gen_test_data()
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
	if (best_prob[i] + 1 == int(test_labels[i])):
		correct_guess[int(test_labels[i]) - 1] = correct_guess[int(test_labels[i]) - 1] + 1

# Find how many of each image our array test_labels has
y_digits = np.zeros((global_num_classes, 1))
for i in range(global_num_classes):
	for j in range(len(test_labels)):
		if (test_labels[j] == i + 1):
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

Wrong1_2 = 0
Wrong1_3 = 0
Wrong1_4 = 0
Wrong2_1 = 0
Wrong2_3 = 0
Wrong2_4 = 0
Wrong3_1 = 0
Wrong3_2 = 0
Wrong3_4 = 0
Wrong4_1 = 0
Wrong4_2 = 0
Wrong4_3 = 0

# Let's see what was mixed up the most
for i in range(len(best_prob)):
	if (best_prob[i] + 1 == 1 and int(test_labels[i]) == 2):
		Wrong1_2 += 1
	elif (best_prob[i] + 1 == 1 and int(test_labels[i]) == 3):
		Wrong1_3 += 1
	elif (best_prob[i] + 1 == 1 and int(test_labels[i]) == 4):
		Wrong1_4 += 1
	elif (best_prob[i] + 1 == 2 and int(test_labels[i]) == 1):
		Wrong2_1 += 1
	elif (best_prob[i] + 1 == 2 and int(test_labels[i]) == 3):
		Wrong2_3 += 1
	elif (best_prob[i] + 1 == 2 and int(test_labels[i]) == 4):
		Wrong2_4 += 1
	elif (best_prob[i] + 1 == 3 and int(test_labels[i]) == 1):
		Wrong3_1 += 1
	elif (best_prob[i] + 1 == 3 and int(test_labels[i]) == 2):
		Wrong3_2 += 1
	elif (best_prob[i] + 1 == 3 and int(test_labels[i]) == 4):
		Wrong3_4 += 1
	elif (best_prob[i] + 1 == 4 and int(test_labels[i]) == 1):
		Wrong4_1 += 1
	elif (best_prob[i] + 1 == 4 and int(test_labels[i]) == 2):
		Wrong4_2 += 1
	elif (best_prob[i] + 1 == 4 and int(test_labels[i]) == 3):
		Wrong4_3 += 1
	
print Wrong1_2, Wrong1_3, Wrong1_4, Wrong2_1, Wrong2_3, Wrong2_4, Wrong3_1, Wrong3_2, Wrong3_4, Wrong4_1, Wrong4_2, Wrong4_3

# Attempting to set up confusion matrix
avg1 = [0 ,0 ,0 ,0]
avg2 = [0 ,0 ,0 ,0]
avg3 = [0 ,0 ,0 ,0]
avg4 = [0 ,0 ,0 ,0]
for i in range(len(prob_all)):
	if (int(test_labels[i]) == 1):
		avg1 += prob_all[i]
	elif (int(test_labels[i]) == 2):
		avg2 += prob_all[i]
	elif (int(test_labels[i]) == 3):
		avg3 += prob_all[i]
	elif (int(test_labels[i]) == 4):
		avg4 += prob_all[i]
avg1 /= y_digits[0]
avg2 /= y_digits[1]
avg3 /= y_digits[2]
avg4 /= y_digits[3]

confuse_mat = np.concatenate(([avg1], [avg2], [avg3], [avg4]), axis = 0)
bar_plot = np.zeros(4)

for i in range(4):
	bar_plot[i] = np.amax(confuse_mat[i])	
print bar_plot

img = plt.imshow(confuse_mat, cmap = 'coolwarm', interpolation = 'none')
plt.colorbar()
#plt.savefig('images/confusionMatrix.png', transparent = True, format = 'png')
plt.show()


