import scipy.io
import numpy as np
import matplotlib.pyplot as plt

####### Neural Networks #######

# Extract the provided data. We need to use scipy since the data is in a matlab file format
data = scipy.io.loadmat('ex3data1.mat')
data_thetas = scipy.io.loadmat('ex3weights.mat')

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

print theta2_vals

# Sigmoid equation
def sigmoid(arr_theta, arr_x):
	return 1.0 / (1.0 + np.exp(-np.dot(arr_x, arr_theta.T)))

# Add a column of ones to our array of x_vals
m = len(x_vals)    # Number of training examples (rows)
arr_ones = np.ones((m, 1))
x_vals = np.hstack((arr_ones, x_vals))

print x_vals.shape

# We will be running our sigmoid equation twice, once for the input and once for the hidden layer
hidden_prob = sigmoid(x_vals, theta1_vals)

# Add a column of ones to our array of x_vals
hidden_prob = hidden_prob.T
arr_ones = np.ones((m, 1))
hidden_prob = np.hstack((arr_ones, hidden_prob))

output_prob = sigmoid(hidden_prob, theta2_vals)

print output_prob

# Find the largest value in each column
best_prob = np.zeros((len(output_prob), 1))
for i in range (len(output_prob)):
	best_prob[i, 0] = np.argmax(output_prob[i, :])
	 
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
