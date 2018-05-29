# Import the necessary pachages
import scipy.io
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

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
The array for theta1 is under the name 'Theta1' and the array for theta2 is under the nam 'Theta2'
theta1_vals is a (25, 401) array and theta2_vals is a (10, 26) array

'''
theta1_vals = data_thetas['Theta1']
theta2_vals = data_thetas['Theta2']

####### Logistic regression #######


# Sigmoid equation
def sigmoid(arr_theta, arr_x):
	return 1.0 / (1.0 + np.exp(-np.dot(arr_x, arr_theta.T)))

# Cost function
def reg_cost(h, arr_x, arr_y):	
	first_half = np.dot(arr_y.T, np.log(h))
	second_half = np.dot((1 - arr_y).T, np.log(1 - h))

	cost = np.sum((-1.0 / m) * (first_half + second_half))
	print cost
	return cost

# Cost function gradient
def reg_cost_gradient(arr_theta, arr_x, arr_y):
	# Scipy.optimize.minimize gives array_theta in 1D, so change it back into a 2D array
	arr_theta = np.reshape(arr_theta, (1, len(arr_theta)))

	h = sigmoid(arr_theta, arr_x).T
	gradient = (1.0 / m) * np.add(np.dot((h - arr_y).T, arr_x), (lambda1 / m) * arr_theta)
	return gradient.flatten()

# Add a column of ones to our array of x_vals
m = len(x_vals)    # Number of training examples (rows)
arr_ones = np.ones((m, 1))
x_vals = np.hstack((arr_ones, x_vals))

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

# We will be running our sigmoid equation twice
a2 = sigmoid(theta1_vals, x_vals)


# Add a column of ones to our array of a2
arr_ones = np.ones((m, 1))
a2 = np.hstack((arr_ones, a2))

print theta2_vals

# Second run thru
a3 = sigmoid(a2, theta2_vals)

J_val = reg_cost(a3, x_vals, y_vals_train)

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

