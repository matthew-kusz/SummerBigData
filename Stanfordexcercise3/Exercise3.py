# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np

# Create arrays of the data given
x_vals = np.genfromtxt("ex3x.dat")
y_vals = np.genfromtxt("ex3y.dat")

# For the normal equation we will need an array of x values that are not normalized
x_vals1 = np.genfromtxt("ex3x.dat")

# Inititalize the number of training examples
m = 47

# Set up the column of ones needed for the x intercept term
arr_ones = np.ones((m,1))

# Now we need to combine the two arrays of columns together
x_vals = np.hstack((arr_ones, x_vals))
x_vals1 = np.hstack((arr_ones, x_vals1))

# Set up your y_vals to have the correct dimensions (currently has dim of (47,))
y_vals = np.resize(y_vals, (len(x_vals), 1))

# Now we need to normalize our inputs so they increase our gradient descent's efficiency
# For living areas
sigma1 = np.std(x_vals[:,1])
mu1 = np.mean(x_vals[:,1])
x_vals[:,1] = (x_vals[:,1] - mu1) / sigma1

# For number of bedrooms
sigma2 = np.std(x_vals[:,2])
mu2 = np.mean(x_vals[:,2])
x_vals[:,2] = (x_vals[:,2] - mu2) / sigma2

# Now we will set up the equations for the hypothesis function, the batch gradient descent update rule, and the cost function
def hyp(arr_theta, arr_x):
	return np.dot(arr_x, arr_theta)

def grad_descent(arr_theta, arr_x, arr_y, alpha):
	h = hyp(arr_theta, arr_x)
	return arr_theta - alpha * (1.0 / m) * np.dot((h - arr_y).T , arr_x).T

def cost(arr_theta, arr_x, arr_y):
	h = hyp(arr_theta, arr_x)
	return (1.0 / (2 * m)) * np.dot((h - arr_y).T, (h - arr_y))	

# Setting up our theta columns
theta_vals = np.array([0, 0, 0])
theta_vals = np.reshape(theta_vals, (len(theta_vals), 1))
theta2_vals = np.array([0, 0, 0])
theta2_vals = np.reshape(theta2_vals, (len(theta2_vals), 1))
theta3_vals = np.array([0, 0, 0])
theta3_vals = np.reshape(theta3_vals, (len(theta3_vals), 1))
theta4_vals = np.array([0, 0, 0])
theta4_vals = np.reshape(theta4_vals, (len(theta4_vals), 1))
theta5_vals = np.array([0, 0, 0])
theta5_vals = np.reshape(theta5_vals, (len(theta5_vals), 1))
theta6_vals = np.array([0, 0, 0])
theta6_vals = np.reshape(theta6_vals, (len(theta6_vals), 1))

# Set up the amount of iterations that we want
iterations = 50

# Set up an array for the cost function (J) to hold a value for each iteration
# Set up multiple J's to compare different alphas
J1_vals = np.zeros(iterations)
J1_vals = np.reshape(J1_vals, (len(J1_vals), 1))
J2_vals = np.zeros(iterations)
J2_vals = np.reshape(J2_vals, (len(J2_vals), 1))
J3_vals = np.zeros(iterations)
J3_vals = np.reshape(J3_vals, (len(J3_vals), 1))
J4_vals = np.zeros(iterations)
J4_vals = np.reshape(J4_vals, (len(J4_vals), 1))
J5_vals = np.zeros(iterations)
J5_vals = np.reshape(J5_vals, (len(J5_vals), 1))
J6_vals = np.zeros(iterations)
J6_vals = np.reshape(J6_vals, (len(J6_vals), 1))

# Set up our inital learning rates
alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3]

# Solve for each J
for i in range(len(J1_vals)):

	# Use gradient decent for theta
	if (i != 0):
		theta_vals = grad_descent(theta_vals, x_vals, y_vals, alpha[0])
		theta2_vals = grad_descent(theta2_vals, x_vals, y_vals, alpha[1])
		theta3_vals = grad_descent(theta3_vals, x_vals, y_vals, alpha[2])
		theta4_vals = grad_descent(theta4_vals, x_vals, y_vals, alpha[3])
		theta5_vals = grad_descent(theta5_vals, x_vals, y_vals, alpha[4])
		theta6_vals = grad_descent(theta6_vals, x_vals, y_vals, alpha[5])

	# Apply new theta values to our cost function
	J1_vals[i] = cost(theta_vals, x_vals, y_vals)
	J2_vals[i] = cost(theta2_vals, x_vals, y_vals)
	J3_vals[i] = cost(theta3_vals, x_vals, y_vals)
	J4_vals[i] = cost(theta4_vals, x_vals, y_vals)
	J5_vals[i] = cost(theta5_vals, x_vals, y_vals)
	J6_vals[i] = cost(theta6_vals, x_vals, y_vals)

# Plot the different alpha values for our cost function
plt.figure(figsize = (8, 6))
plt.plot(J1_vals, 'r', J2_vals, 'g', J3_vals, 'k', J4_vals, 'b', J5_vals, 'm', J6_vals, 'c')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Cost Function with Different Alpha Values')
plt.legend(['Alpha = 0.01', 'Alpha = 0.03', 'Alpha = 0.1', 'Alpha = 0.03', 'Alpha = 1', 'Alpha = 1.3'], prop = {'size':10}, loc = 9) 
plt.show()

# Our best learning rate was alpha = 1, so now we will find the final values of theta for it
# We will use gradient decent until it converges
for i in range(100):
	theta5_vals = grad_descent(theta5_vals, x_vals, y_vals, alpha[4])

print "The values for theta are theta0 = %g, theta1 = %g, theta2 = %g" %(theta5_vals[0], theta5_vals[1], theta5_vals[2])

# Predict the value of a 1650-square-foot house with 3 bedrooms, remember to scale the values
print "The value of a 1650-square-foot house with 3 bedrooms is %g dollars." %(theta5_vals[0] + theta5_vals[1] * ((1650 - mu1) / sigma1) + theta5_vals[2] * ((3 - mu2) / sigma2))

# Setting up our normal equation
def normal(arr_x, arr_y):
	first_half = np.dot(arr_x.T, arr_x)
	second_half = np.dot(arr_x.T, arr_y)
	first_half = np.linalg.pinv(first_half)
	return np.dot(first_half, second_half)

# Set up theta
theta = np.array([0, 0, 0])
theta = np.reshape(theta, (len(theta), 1))

# Solve for theta
theta = normal(x_vals1, y_vals)
print "The values for theta are theta0 = %g, theta1 = %g, theta2 = %g" %(theta[0], theta[1], theta[2])

# Predict the value of a 1650-square-foot house with 3 bedrooms, again
print "The value of a 1650-square-foot house with 3 bedrooms is %g dollars." %(theta[0] + theta[1] * 1650 + theta[2] * 3)
