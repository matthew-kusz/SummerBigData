import numpy as np
import matplotlib.pyplot as plt

# Set up the data
train500 = [93.5, 96.7, 84.2, 82.4, 85.3, 70.7, 88.7, 88.6, 76.2, 78.2]
train5000 = [97.6, 98.2, 90.5, 90.2, 95, 88.1, 94.2, 92.3, 89.1, 90.7]
train10000 = [98.1, 98.1, 93, 93, 95.3, 89.6, 95.1, 93.6, 92.9, 91.1]
train20000 = [97.3, 98.3, 94.6, 93.1, 96, 92.5, 96.5, 94.2, 93.3, 92.8]
train40000 = [98.3, 98.5, 95.2, 93.7, 96.7, 92, 97.2, 94.7, 94.1, 94]
train60000 = [98.2, 98.8, 95.3, 95, 96.2, 92.4, 96.8, 95.2, 94.8, 93.8]

# For printing out the results in a bar graph
x = [0, 1, 2 ,3 ,4 ,5 ,6 ,7 , 8 ,9]
x = np.reshape(x, (len(x), 1))
fig1 = plt.figure()
plt.bar(x - 0.35, train500, width = 0.14, align = 'center', color = 'goldenrod')
plt.bar(x - 0.21, train5000, width = 0.14, align = 'center', color = 'blue')
plt.bar(x - 0.07, train10000, width = 0.14, align = 'center', color = 'green')
plt.bar(x + 0.07, train20000, width = 0.14, align = 'center', color = 'black')
plt.bar(x + 0.21, train40000, width = 0.14, align = 'center', color = 'purple') 
plt.bar(x + 0.35, train60000, width = 0.14, align = 'center', color = 'firebrick')
plt.title("Accuracy with Different Training Sizes (1 hour run time)")
plt.xlim(left = -0.5, right = 9.5)
plt.ylim(70)
plt.xlabel("Number")
plt.ylabel('Percentage of correct guesses')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.legend([500, 5000, 10000, 20000, 40000, 60000], loc = 'upper right', ncol = 6, prop = {'size': 9})
plt.show()

# Plot the average correct guesses for all 6 cases
x2 = [1, 2, 3, 4, 5, 6]
averages = [84.45, 92.69, 93.98, 94.86, 95.44, 95.58]
fig2 = plt.figure()
plt.bar(x2, averages, align = 'center', color = 'red')
plt.title('Average Accuracy for Different Training Sizes')
plt.ylim(80)
plt.xlabel('Training size')
plt.xticks(x2, [500, 5000, 10000, 20000, 40000, 60000])
plt.ylabel('Average accuracy')
plt.show()

# Plot with different lambdas for 60,000 training set
x3 = [1, 2, 3, 4]
averages2 = [95.58, 95.64, 95.8, 93.19]
plt.bar(x3, averages2, align = 'center', color = 'red')
plt.title('Accuracy Using 60,000 Training Set with Different Lambda Values (1 hour)')
plt.ylim(90)
plt.xlabel('Lambda')
plt.xticks(x3, [0.1, 1, 10, 100])
plt.ylabel('Average accuracy')
plt.show()

# Plot probabilities using 60,000 training set that was rain for 3 hours
x4 = [0, 1, 2, 3, 4, 5, 6, 7 ,8, 9]
train60000_3hours = [97.9, 98.8, 94.5, 95.4, 95.8, 93.9, 96.5, 94.4, 94.5, 93.6]
fig4 = plt.figure()
plt.bar(x4, train60000_3hours, align = 'center', color = 'red')
plt.title('Accuracy Using 60,000 Training Set Ran for 3 Hours')
plt.xlim(left = -0.5, right = 9.5)
plt.ylim(90)
plt.xlabel('Number')
plt.ylabel('Percentage of correct guesses')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.show()