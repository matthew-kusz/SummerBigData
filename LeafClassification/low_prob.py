# Import the necessary packages
import numpy as np				 	 # Allow for easier use of arrays and linear algebra
import data_setup					 # Python code for setting up the data
import visualize					 # Python code for visualizing images
import pandas as pd                         	 	 # For reading in and writing files
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler 	 # Preprocessing
from sklearn.model_selection import GridSearchCV	 # Helps find the best parameters for our model

####### Global variables #######
global_max_dim = 50
global_num_classes = 99
threshold = 0.95
filename = 'Sklearn_all_pred.npy'

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
train, test = data_setup.more_features(train, test, train_list, test_list)

# fit calculates the mean and transform centers and scales the data so we have 0 mean and unit variance
scaler = StandardScaler().fit(train)
x_train = scaler.transform(train)
x_test = scaler.transform(test)

y_pred = np.load(filename)

# Look at some of the lower probability leaves
visualize.confusion(y_pred, y, classes, test_ids, global_num_classes, train_mod_list, threshold)
