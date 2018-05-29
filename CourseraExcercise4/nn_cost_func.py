# Import the necessary pachages
import scipy.io
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Regularized cost function
def reg_cost(h, arr_x, arr_y, theta1, theta2, lambda1):	
	first_half = np.multiply(arr_y, np.log(h))
	second_half = np.multiply((1 - arr_y), np.log(1 - h))

	cost1 = np.sum((-1.0 / m) * (first_half + second_half))
	cost2 = (lambda1 / (2.0 * m)) * (np.sum(theta1 ** 2) - np.sum(theta1[:,0] ** 2))
	cost3 = (lambda1 / (2.0 * m)) * (np.sum(theta2 ** 2) - np.sum(theta2[:,0] ** 2))
	cost = cost1 + cost2 + cost3
	return cost

# Regularized cost function gradient
def reg_cost_gradient(arr_theta, arr_x, arr_y):
	# Scipy.optimize.minimize gives array_theta in 1D, so change it back into a 2D array
	arr_theta = np.reshape(arr_theta, (1, len(arr_theta)))

	h = sigmoid(arr_theta, arr_x).T
	gradient = (1.0 / m) * np.add(np.dot((h - arr_y).T, arr_x), (lambda1 / m) * arr_theta)
	return gradient.flatten()

# Feedforward
def feedforward(theta1, theta2, arr_x):
	# We will be running our sigmoid function twice
	a2 = sigmoid(np.dot(arr_x, theta1.T))
	
	# Add a column of ones to our array of a2
	arr_ones = np.ones((m, 1))
	a2 = np.hstack((arr_ones, a2))
	
	# Second run thru
	a3 = sigmoid(np.dot(a2, theta2.T))

	return a3

# Sigmoid function gradient
def sigmoid_gradient(value):
	h = sigmoid(value)
	return h * (1 - h)

# Backpropagation
####### Neural Network #######
'''
def Backprop(theta_rand, arr_x, arr_y, lambda1):
	
	for i in size(m):
		
	return
'''	

# Extract the provided data. We need to use scipy since the data is in a matlab file format
data = scipy.io.loadmat('ex4data1.mat')
data_thetas = scipy.io.loadmat('ex4weights.mat')

'''
The x array is under the name 'X', the y array is under the name 'y'
x_vals ia a (5000, 400) array and y_vals is a (5000, 1) array
'''
x_vals = data['X']
y_vals = data['y']

'''
The array for theta1 is under the name 'Theta1' and the array for theta2 is under the name 'Theta2'
theta1_vals is a (25, 401) array and theta2_vals is a (10, 26) array

'''
theta1_vals = data_thetas['Theta1'] # FIXME
theta2_vals = data_thetas['Theta2'] # FIXME

#Set what lambda value we want to use
lambda1 = 1

# Add a column of ones to our array of x_vals
m = len(x_vals)    # Number of training examples (rows)
arr_ones = np.ones((m, 1))
x_vals = np.hstack((arr_ones, x_vals))        # (5000, 401) matrix

# Set up an array that will be either 1 or 0 depending on which number we are looking at
y_vals_train = np.zeros((len(y_vals), 10))

for i in range(10):
	# Set up an array with the values that stand for each number (10 stands for 0)
	arr_num = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	
	for j in range(len(y_vals)):
		if (y_vals[j] == arr_num[i]):
			y_vals_train[j, i:] = 1
		
		else:
			y_vals_train[j, i:] = 0

'''
For testing cost and feedforward
hypothesis = feedforward(theta1_vals, theta2_vals, x_vals)
J_val = reg_cost(hypothesis, x_vals, y_vals_train, theta1_vals, theta2_vals, lambda1)
J_val = 10.5
This differs from the value in the intruction sheet
'''

####### Backpropagation ########

# Randomly initialize our theta values in a range [-0.12, 0.12]
n = len(x_vals[0])   # Number of columns
random_theta1 = np.random.rand(25, n)                    # (25, 401) matrix
random_theta1 = random_theta1 * 2 * 0.12 - 0.12

random_theta2 = np.random.rand(10, len(random_theta1) + 1)   # (10, 26) matrix
random_theta2 = random_theta2 * 2 * 0.12 - 0.12

'''	
	# Use scipys minimize function to compute the theta values
	minimum = scipy.optimize.minimize(fun = reg_cost, x0 = theta_vals, method = 'BFGS', jac = reg_cost_gradient, args = (x_vals, y_vals_train, lambda_all[k]))#, options = {'disp': True}
	
	# Create a new array that consists of all of the new theta values
	theta_new = np.reshape(minimum.x, (1, len(minimum.x)))
	prob = sigmoid(theta_new, x_vals)
	prob = np.reshape(prob, (len(prob), 1))
	
	if (i == 0):
		prob_all = prob
	else:
		prob_all = np.hstack((prob_all, prob))	

	print "Iteration: %g" %(i + 1)

# Find the largest value in each column
best_prob = np.zeros((len(prob_all), 1))
for i in range (len(prob_all)):
	best_prob[i, 0] = np.argmax(prob_all[i, :])
	 
# Find how accurate our program was with identifying the correct number
correct_guess = np.zeros((10, 1))
for i in range(len(best_prob)):
	if (best_prob[i] == y_vals[i]):
		correct_guess[y_vals[i]] = correct_guess[y_vals[i]] + 1
	
	if (best_prob[i] == 0 and y_vals[i] == 10):
		correct_guess[0] = correct_guess[0] + 1
	
# Calculate the percentage
correct_guess = (correct_guess / 500) * 100

# Check the results
print correct_guess

correct_guesses[k] = correct_guess
'''

