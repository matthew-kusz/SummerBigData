# Import the necessary packages
import numpy as np
import pandas as pd
import data_setup

####### Global variables #######
num_probs = 10
filename = 'sklearn_log_reg.csv'
threshold = 0.95

####### Code #######
# Set up the data given to us
train_list, test_list, train_ids, test_ids, train, test, y, y_train, classes = data_setup.data()

# Load up all of the probabilities
probs = []
for i in range(num_probs):
	probs.append(np.load('probabilities/sklearn' + str(i + 1) +'.npy'))

# Find the average probabilities
all_pred = np.zeros((probs[0].shape[0], probs[0].shape[1]))
for i in range(len(probs)):
	all_pred += probs[i]

all_pred /= 10

# Save the average probabilities to a .npy file
np.save('Sklearn_all_pred', all_pred)

# Let's see what leaves our network struggled the most with
# visualize.confusion(probs, y, classes, test_ids, global_num_classes, train_mod_list, threshold)

# Set up the predictions into the correct format to submit to Kaggle
all_pred = pd.DataFrame(all_pred, index = test_ids, columns = classes)

# Save predictions to a csv file to submit
print 'Saving to file', filename, '...'
fp = open(filename,'w')
fp.write(all_pred.to_csv())

print 'Finished.'
