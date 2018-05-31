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

	# For visualizing the hidden layer
	a2_visual = a2	

	# Add a column of ones to our array of a2
	arr_ones = np.ones((m, 1))
	a2 = np.hstack((arr_ones, a2))        # (5000, 26) matrix

	# Second run
	a3 = sigmoid(np.dot(a2, theta2.T))    # (5000, 10) matrix

	return a3, a2_visual


# Import the theta valus that we found earlier
theta = np.genfromtxt("finalThetasRandom5000.out")
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
prob_all, a2_v = feedforward(theta_vals1, theta_vals2, x_vals)

# Let's visualize a2 now
# Set up an array thet holds the images
image = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]

# Set up whitespace to help with visualizing the hidden layer
white_space = np.zeros((5,5))

# Plotting 0 to 9
for i in range (len(image)):
	temp = np.reshape(a2_v[i * 500], (5, 5))
	temp = temp.T
	image[i] = temp

# Put all the images together
total_images = np.concatenate((image[0], white_space, image[1], white_space, image[2], white_space, image[3], white_space,image[4], white_space, image[5], white_space, image[6], white_space,image[7], white_space,image[8], white_space,image[9], white_space), axis = 1)

# Plot the results
plt.imshow(total_images, cmap = 'binary')
plt.show()

# Let's look at what theta_vals1 looks like
# First we need to remove the bias layer
theta_vals1_modified = theta_vals1[:, 1:401]

# Set up an array thet holds the images
image2 = image

# Set up whitespace to help with visualizing theta1
white_space2 = np.zeros((20,20))

# Plotting 0 to 9
for i in range (len(image2)):
	temp = np.reshape(theta_vals1_modified[i], (20, 20))
	temp = temp.T
	image2[i] = temp

# Put all the images together
total_images2 = np.concatenate((image2[0], white_space2, image2[1], white_space2, image2[2], white_space2, image2[3], white_space2, image2[4], white_space2, image2[5], white_space2, image2[6], white_space2, image2[7], white_space2, image2[8], white_space2, image2[9], white_space2), axis = 1)

# Plot the results
plt.imshow(image2[0], cmap = 'binary')
plt.show()

# Set up an array thet holds the images
image3 = image

# Plotting 10 to 19
for i in range (len(image2)):
	temp = np.reshape(theta_vals1_modified[i + 10], (20, 20))
	temp = temp.T
	image3[i] = temp

total_images3 = np.concatenate((image3[0], white_space2, image3[1], white_space2, image3[2], white_space2, image3[3], white_space2, image3[4], white_space2, image3[5], white_space2, image3[6], white_space2, image3[7], white_space2, image3[8], white_space2, image3[9], white_space2), axis = 1)

# Set up an array thet holds the images
image4 = image

# Plotting 20 to 25
for i in range (5):
	temp = np.reshape(theta_vals1_modified[i + 20], (20, 20))
	temp = temp.T
	image4[i] = temp

total_images4 = np.concatenate((image4[0], white_space2, image4[1], white_space2, image4[2], white_space2, image4[3], white_space2, image4[4], white_space2, white_space2, white_space2, white_space2, white_space2, white_space2, white_space2, white_space2, white_space2, white_space2, white_space2), axis = 1)

combined_images = np.concatenate((total_images2, total_images3, total_images4), axis = 0)

plt.imshow(combined_images, cmap = 'binary')
plt.show()

# Now lets visual our theta2
theta_vals1_modified = theta_vals1[:, 1:26]

image5 = image

# Plotting 0 to 9
for i in range (len(image5)):
	temp = np.reshape(theta_vals1_modified[i], (5, 5))
	temp = temp.T
	image2[i] = temp

total_images5 = np.concatenate((image5[0], white_space, image5[1], white_space, image5[2], white_space, image5[3], white_space,image5[4], white_space, image5[5], white_space, image5[6], white_space,image5[7], white_space,image5[8], white_space,image5[9], white_space), axis = 1)

plt.imshow(image5[0], cmap = 'binary')
plt.show()
