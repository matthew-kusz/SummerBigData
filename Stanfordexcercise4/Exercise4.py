# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt

# Extract the data given
x_vals = np.genfromtxt("ex4x.dat")
y_vals = np.genfromtxt("ex4y.dat")

"""
x-vals has two columns, the first column consists of scores from test 1 and the second column
consists of scores from test 2. y_vals is an array that has the numbers 1 and 0. 1 being for
students who were admitted and 0 representing those who weren't.
"""

m = len(x_vals)  # Set up the number of training examples

# Set up a column of ones for our x intercept to attach to our column of x values
arr_ones = np.ones((m, 1))

# Attach the column of ones and column of x's together
x_vals = np.hstack((arr_ones, x_vals))

# Reshape the y_vals array intoa column
y_vals = np.reshape(y_vals, (m, 1))


# Set up two arrays for when y is either 1 or 0
y_one1 = []
y_one2 = []
y_zero1 = []
y_zero2 = []

for i in range(len(x_vals)):
	if (y_vals[i] == 1):
		y_one1 = np.append(y_one1, x_vals[i, 1])
		y_one2 = np.append(y_one2, x_vals[i, 2])

	else:
		y_zero1 = np.append(y_zero1, x_vals[i, 1])
		y_zero2 = np.append(y_zero2, x_vals[i, 2])

y_one1 = np.reshape(y_one1, (len(y_one1), 1))
y_one2 = np.reshape(y_one2, (len(y_one2), 1))
y_zero1 = np.reshape(y_zero1, (len(y_zero1), 1))
y_zero2 = np.reshape(y_zero2, (len(y_zero2), 1))

y_one = np.hstack((y_one1, y_one2))
y_zero = np.hstack((y_zero1, y_zero2))

# Create a plot of the data provided.
plt.plot(y_one[:, 0], y_one[:, 1], 'o', y_zero[:, 0], y_zero[:, 1], '^')
plt.xlabel("Exam 1 Scores")
plt.ylabel("Exam 2 scores")
plt.legend(["Admitted", "Not admitted"], prop={'size':10})
plt.title("People Admitted into College Vs. People Not Admitted")
plt.show()

# Now we need to set up our equations
# Sigmoid equation
def sigmoid(arr_theta, arr_x):
	return 1 / (1.0 + np.exp(-np.dot(arr_x, arr_theta)))

# Cost function
def cost(arr_theta, arr_x, arr_y):
	h = sigmoid(arr_theta, arr_x)
	first_half = np.dot(arr_y.T, np.log(h))
	second_half = np.dot((1 - arr_y).T, np.log(1 - h))
	return (-1.0 / m) * (first_half + second_half)

# Cost function gradient
def cost_gradient(arr_theta, arr_x, arr_y):
	return (1.0 / m) * np.dot((sigmoid(arr_theta, arr_x) - arr_y).T, arr_x)

# Hessian
def hessian(arr_theta, arr_x):
	h = sigmoid(arr_theta, arr_x)
	first_half = np.dot(arr_x.T, np.diag(h[:,0]))
	second_half = np.dot(np.diag((1 - h)[:,0]), arr_x)
	return	(1.0 / m) * np.dot(first_half, second_half)

# Newton's Method
def newton_method(arr_theta, arr_x, arr_y):
	gradient = cost_gradient(arr_theta, arr_x, arr_y)
	Hessian = hessian(arr_theta, arr_x)
	Hessian_inverse = np.linalg.pinv(Hessian)
	return gradient, arr_theta - np.dot(Hessian_inverse, gradient.T)

# Set up our theta vector
theta_vals = np.array([0, 0, 0])
theta_vals = np.reshape(theta_vals, (len(theta_vals), 1))

max_iterations = 10 # Max number of iterations

# Set up an empty array for the values of our cost function
J_vals = []

gradient = 1  # Gradient of the cost
i = 0         # What iteration the loop is on
while (abs(np.sum(gradient)) > 1e-5 and i < max_iterations):

	if (i != 0):
		# Use Newton's method to find the values of J
		gradient, theta_vals = newton_method(theta_vals, x_vals, y_vals)
	
	# Add each new value to the end of the array
	J_vals = np.append(J_vals, cost(theta_vals, x_vals, y_vals))

	i += 1

# Plot the cost function
plt.plot(J_vals, '-o')
plt.xlabel('Iteration')
plt.ylabel('Cost (J)')
plt.title('Cost Function Value Vs. Iteration')
plt.show()

# Print the results
print "My theta values are: Theta0 = %g, Theta1 = %g, Theta2 = %g. I needed %g iterations for convergence." %(theta_vals[0], theta_vals[1], theta_vals[2], i)

# Now we have to find the decision boundary
# Find the leftmost and rightmost x values on the horizontal axis
x_max = max(x_vals[:,1])
x_min = min(x_vals[:,1])

# Find the associated y values
y_left = (-theta_vals[0] - theta_vals[1] * x_min) / theta_vals[2]
y_right = (-theta_vals[0] - theta_vals[1] * x_max) / theta_vals[2]

x_coords = [x_max, x_min]
y_coords = [y_right, y_left]

# Plot the exam scores along with the decision boundary line
plt.plot(y_one[:, 0], y_one[:, 1], 'o', y_zero[:, 0], y_zero[:, 1], '^', x_coords, y_coords, 'r')
plt.xlabel("Exam 1 Scores")
plt.ylabel("Exam 2 scores")
plt.legend(["Admitted", "Not admitted", "Decision boundary"], prop={'size':8})
plt.title("People Admitted into College Vs. People Not Admitted")
plt.show()

# Find the probability that a student with a score of 20 on Exam 1 and a score of 80 on Exam 2 will not be admitted
test_results = ([1, 20, 80])
test_results = np.reshape(test_results, (1, 3))
admission = sigmoid(theta_vals, test_results)
print "The probability that a student with a score of 20 on Exam 1 and a score of 80 on Exam 2 will not be admitted is %g." %(1 - admission)
