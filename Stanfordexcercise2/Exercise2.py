# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Create an array for the x and y values from the files given
x_vals = np.genfromtxt("ex2x.dat")
y_vals = np.genfromtxt("ex2y.dat")

#For the second plot
x_vals1 = x_vals
y_vals1 = y_vals

# Plot the x and y values
fig1 = plt.figure(figsize = (8, 6))
plt.plot(x_vals, y_vals, 'o')
plt.title('Height Vs. Age')
plt.ylabel('Height in meters')
plt.xlabel('Age in years')
plt.legend(['Training Data'], prop={'size':10})

# Set up the number of training examples 
m = len(x_vals)

# Set up the column of ones needed from the x0 intercept
arr_ones = np.ones((m, 1))

# Adjust x_vals array so it is the same dimensions as arr_ones
x_vals = np.reshape(x_vals, (m, 1))

# Add this column to the array of x_vals
x_vals = np.hstack((arr_ones, x_vals))

# Set up the learning rate
alpha = 0.07

# Creating the a column vector with theta0 and theta1 in it
arr_theta = np.array([0, 0])
arr_theta = np.reshape(arr_theta, (len(arr_theta), 1))

# Set the y_vals to be the same dimensions as x_vals
y_vals = np.reshape(y_vals, (m, 1))

# Set up the equations for linear regression and gradient descent
 
# linear regression
def Lin_reg(arr_theta, arr_x):
    return np.dot(arr_x, arr_theta)

# Gradient descent
def grad_reg(arr_theta, arr_x, arr_y):
    
    # Initialize the necessary variables
    summ = 0
    gradient_cost = 0
    h_theta = 0
    
    # Solve for h_theta
    h_theta = Lin_reg(arr_theta, arr_x)
    
    # Calculate the gradient of the cost
    summ = np.dot((h_theta - arr_y).T, arr_x)
    gradient_cost = (1.0 / m) * summ
    return gradient_cost, arr_theta.T - alpha * gradient_cost
    
# Solving for theta0 and theta1
# Set up the necessary parameters to find when theta converges
iterations = 2000
change= 0.00001
i = 1
gradient = 0

gradient, arr_theta = grad_reg(arr_theta, x_vals, y_vals)
arr_theta = arr_theta.T
print arr_theta	
while (i < iterations and abs(np.sum(gradient)) > change):
    # Change theta0 and theta1 simultaneously
    gradient, arr_theta = grad_reg(arr_theta, x_vals, y_vals)
    
    # Transpose arr_theta to get it back into the right dimensions
    arr_theta = arr_theta.T
        
    i += 1

print('Our values are: theta0 = %g, theta1 = %g.' %(arr_theta[0], arr_theta[1]))

# We will create our line to plot
arr_line = [0 for i in range(len(x_vals))]

for i in range(len(x_vals)):
    arr_line[i] = arr_theta[0] + arr_theta[1] * x_vals[i]
    

fig2 = plt.figure(figsize = (8, 6))
plt.plot(x_vals1, y_vals1, 'o', x_vals, arr_line, 'r')
plt.title('Height Vs. Age with Linear Regression After 2000 Iterations')
plt.ylabel('Height in meters')
plt.xlabel('Age in years')
plt.legend(['training Data', 'Linear Regression'], prop={'size':10})
plt.show()

# We are now going to create a surface plot for our cost function
# Initialize J_vals to 100x100 matrix of 0's
J_vals = np.zeros((100,100))
theta0_vals = np.linspace(-3, 3, 100)
theta1_vals = np.linspace(-1, 1, 100)

# Find the different J values
for i in range(len(theta0_vals)):
	for j in range(len(theta1_vals)):
		t = np.array([theta0_vals[i], theta1_vals[j]])
		t = np.reshape(t, (len(t), 1))
		J_vals[i, j] = (1.0 / (2 * m)) * np.sum((Lin_reg(t, x_vals) - y_vals) ** 2)

# Transpose J_vals so it looks like the provided plot
J_vals = J_vals.T

# Create the surface plot
fig3 = plt.figure()
theta0m, theta1m = np.meshgrid(theta0_vals, theta1_vals)
ax = Axes3D(plt.gcf())
surface = ax.plot_surface(theta0m, theta1m, J_vals, cmap = cm.coolwarm)
plt.ylabel('Theta 1')
plt.xlabel('Theta 0')
plt.title('Cost')
fig3.colorbar(surface, shrink = 0.5, aspect = 16)
plt.show()

# Create the contour plot
fig4 = plt.figure()
contour = plt.contour(theta0_vals, theta1_vals, J_vals)
plt.ylabel('Theta 1')
plt.xlabel('Theta 0')
plt.title('Cost')
fig4.colorbar(contour, shrink = 0.5, aspect = 16)
plt.show()
