# Import the necessary pachages
import scipy.io
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Feedforward
def feedforward(theta1, theta2, arr_x):

	# We will be running our sigmoid function twice
	a2 = sigmoid(np.dot(arr_x, theta1.T))	

	# Add a column of ones to our array of a2
	arr_ones = np.ones((m, 1))
	a2 = np.hstack((arr_ones, a2))        # (5000, 26) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, theta2.T))    # (5000, 10) matrix

	return a3


# Import the theta valus that we found earlier
theta = np.genfromtxt("finalThetasOrdered5000.out")
theta_vals1 = np.reshape(theta[0:25 * 401], (25, 401))
theta_vals2 = np.reshape(theta[25 * 401: len(theta)], (10, 26))

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
# Randomize our 5000 digits
xy_vals = np.concatenate((x_vals, y_vals), axis = 1)
np.random.shuffle(xy_vals)

# Split xy_vals apart back into x and y arrays
x_vals = xy_vals[:, :-1]
y_vals = xy_vals[:, [400]]
'''

# Add a column of ones to our array of x_vals
m = len(x_vals)                               # Number of training examples (rows)
arr_ones = np.ones((m, 1))
x_vals = np.hstack((arr_ones, x_vals))        # (5000, 401) matrix

# Find the probabilities for each digit
prob_all = feedforward(theta_vals1, theta_vals2, x_vals)

# Find the largest value in each column
best_prob = np.zeros((len(prob_all), 1))
for i in range (len(prob_all)):
	best_prob[i, 0] = np.argmax(prob_all[i, :])
	 
# Find how accurate our program was with identifying the correct number
correct_guess = np.zeros((10, 1))
for i in range(len(best_prob)):
	if (best_prob[i] == int(y_vals[i])):
		correct_guess[int(y_vals[i])] = correct_guess[int(y_vals[i])] + 1
	
	if (best_prob[i] == 0 and int(y_vals[i]) == 10):
		correct_guess[0] = correct_guess[0] + 1
	
# Calculate the percentage
correct_guess = (correct_guess / 500) * 100

# Check the results
print correct_guess

