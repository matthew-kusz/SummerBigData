import numpy as np				 	 # Allow for easier use of arrays and linear algebra
import data_setup
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 	 # Preprocessing

global_max_dim = 50

####### Code #######
# Set up the data given to us
train_list, test_list, train_ids, test_ids, train, test, y, y_train, classes = data_setup.data()

# Grab more features to train on
train, test = data_setup.engineered_features(train, test, train_list, test_list)

# We need to reshape our images so they are all the same dimensions
train_mod_list = data_setup.reshape_img(train_list, global_max_dim)
test_mod_list = data_setup.reshape_img(test_list, global_max_dim)

train, test = data_setup.apply_PCA(train, test, train_mod_list, test_mod_list, global_max_dim)

# data_setup.more_features(train, test, train_list, test_list)

# fit_transform() calculates the mean and std and also centers and scales data
x_train = StandardScaler().fit_transform(train)
x_test = StandardScaler().fit_transform(test)

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
log_reg = LogisticRegression(solver = 'lbfgs')

# FIT IT
log_reg.fit(x_train, y_train)

# Predict for One Observation (image)
logisticRegr.predict(x_test)

# Set up the predictions into the correct format to submit to Kaggle
y_pred = pd.DataFrame(y_pred, index = test_ids, columns = classes)

# Save predictions to a csv file to submit
if args.save_prob:
	print 'Saving to file', filename3, '...'
	fp = open(filename3,'w')
	fp.write(y_pred.to_csv())

print 'Finished.'
