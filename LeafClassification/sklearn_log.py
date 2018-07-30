# Import the necessary packages
import numpy as np				 	 # Allow for easier use of arrays and linear algebra
import data_setup
import pandas as pd                         	 	 # For reading in and writing files
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler 	 # Preprocessing
from sklearn.model_selection import GridSearchCV

####### Global variables #######
global_max_dim = 50
filename = 'sklearn_log_reg.csv'

# Set up a seed so that our results don't fluctuate over multiple tests
np.random.seed(1)

####### Code #######
# Set up the data given to us
train_list, test_list, train_ids, test_ids, train, test, y, y_train, classes = data_setup.data()

# Grab more features to train on
train, test = data_setup.engineered_features(train, test, train_list, test_list)

# We need to reshape our images so they are all the same dimensions
train_mod_list = data_setup.reshape_img(train_list, global_max_dim)
test_mod_list = data_setup.reshape_img(test_list, global_max_dim)

# Let's apply PCA to the images and attach them to the pre-extracted features
train, test = data_setup.apply_PCA(train, test, train_mod_list, test_mod_list, global_max_dim)

# data_setup.more_features(train, test, train_list, test_list)

# fit_transform() calculates the mean and std and also centers and scales data
x_train = StandardScaler().fit_transform(train)
x_test = StandardScaler().fit_transform(test)

# We will use the solver 'lbfgs'
log_reg = LogisticRegression(solver = 'lbfgs', verbose = 1, max_iter = 500, multi_class = 'multinomial')

# Find the best parameters for our model
gsCV = GridSearchCV(log_reg, param_grid = {'C': [10000.0], 'tol': [0.00001]}, scoring = 'neg_log_loss', refit = True, verbose = 1, n_jobs = -1, cv = 5)

# FIT IT
gsCV.fit(x_train, y)

print('best params: ' + str(gsCV.best_params_))
for params, mean_score, scores in gsCV.grid_scores_:
 	print '%0.3f (+/-%0.03f) for %r' % (mean_score, scores.std(), params)
 	print scores

# Predict for One Observation (image)
y_pred = gsCV.predict_proba(x_test)

# Set up the predictions into the correct format to submit to Kaggle
y_pred = pd.DataFrame(y_pred, index = test_ids, columns = classes)

# Save predictions to a csv file to submit

print 'Saving to file', filename, '...'
fp = open(filename,'w')
fp.write(y_pred.to_csv())

print 'Finished.'
