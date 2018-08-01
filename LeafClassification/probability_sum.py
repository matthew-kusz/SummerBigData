# Import the necessary packages
import numpy as np
import pandas as pd

num_probs = 10
filename = 'sklearn_log_reg.csv'

probs = []

for i in range(num_probs):
	probs.append(np.load('probabilities/sklearn' + str(i + 1) +'.npy'))

all_pred = np.zeros((probs[0].shape[0], probs[0].shape[1]))
for i in range(len(probs)):
	all_pred += probs[i]

all_pred /= 10
np.save('Sklearn_all_pred', all_pred)	
# Let's see what leaves our network struggled the most with
# visualize.confusion(probs, y, classes, test_ids, global_num_classes, train_mod_list)

# Set up the predictions into the correct format to submit to Kaggle
all_pred = pd.DataFrame(all_pred, index = test_ids, columns = classes)

# Save predictions to a csv file to submit
if args.save_prob:
	print 'Saving to file', filename3, '...'
	fp = open(filename,'w')
	fp.write(all_pred.to_csv())

print 'Finished.'
