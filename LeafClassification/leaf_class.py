# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation
import matplotlib.image as mpimg       # reading images to numpy arrays
import scipy.ndimage as ndi            # finding the center of the leaves

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

####### Global Variables #######
global_num_train = 990
global_num_test = 594
global_input_layer = 192
global_hidden_layer1 = 350
global_hidden_layer2 = 100
global_output_layer = 99
global_epochs = 10
global_batch_size = 10
global_num_classes = 99

####### Definitions #######
def visualize():
	# Using scipy to find the center of the leaf
	img = mpimg.imread('data_provided/images/10.jpg')
	center_y, center_x = ndi.center_of_mass(img)
	
	plt.imshow(img, cmap = 'binary')
	plt.scatter(center_x, center_y)
	plt.show()

	return

# Test the accuracy of our model
def accuracy(model, arr_test, arr_labels):

	prediction = model.predict(arr_test)
	prediction = np.array(np.argmax(prediction, axis = 1))
	accur = np.array([1 for (a,b) in zip(prediction, arr_labels) if a==b ]).sum() / (global_num_test + 0.0)

	return accur, prediction

####### Code #######
# We need to extract the data given
# Set up our training data
train = pd.read_csv('data_provided/train.csv')

# Extract the species of each leaf
y_raw = train.pop('species')

# Label each species from 0 - n-1 and set up a one-hot scheme
le = LabelEncoder()
y = le.fit(y_raw).transform(y_raw)
classes = le.classes_
Y_train = np_utils.to_categorical(y)

# Extract the id of each leaf
train_ids = train.pop('id')

# Set up our testing data
test = pd.read_csv('data_provided/test.csv')

# Extract the id of each leaf
test_ids = test.pop('id')

X_train = StandardScaler().fit(train).transform(train)
X_test = StandardScaler().fit(test).transform(test)

# visualize()

# Setting up the Keras neural network
# Create our model (currently has 1 hidden layer and using softmax regression)
model = Sequential()

model.add(Dense(global_hidden_layer1, input_dim = global_input_layer, activation = 'relu'))
model.add(Dense(global_hidden_layer2, activation = 'relu'))
model.add(Dense(global_output_layer, activation = 'softmax'))

# Compile our model
model.compile(optimizer = 'SGD', loss='categorical_crossentropy', metrics=['accuracy'])
print model.summary()

# Fit our model
# early_stopper=EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
model.fit(X_train, Y_train, epochs= global_epochs, batch_size = global_batch_size)

# Evaluate our model
scores = model.evaluate(X_train, Y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Let's see how accurace our model is
y_pred = model.predict_proba(X_test)
y_pred = pd.DataFrame(y_pred,index=test_ids,columns=classes)

fp = open('submission.csv','w')
fp.write(y_pred.to_csv())
print('finished.')

