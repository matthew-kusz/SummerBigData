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
global_total_img = global_num_train + global_num_test
global_input_layer = 192
global_hidden_layer1 = 350
global_hidden_layer2 = 100
global_output_layer = 99
global_epochs = 400 # 200
global_batch_size = 10
global_num_classes = 99

# Set up a seed so that our results don't fluctuate over multiple tests
np.random.seed(1)

####### Definitions #######
# Look at what some of the leaves look like
def visualize(images):
	#center_y, center_x = ndi.center_of_mass(img)
	
	plt.imshow(images[0], cmap = 'binary')
	#plt.scatter(center_x, center_y)
	#plt.show()

	return

# Set up all of the images in a matrix
def grab_images():
	
	matrix = np.zeros((2, global_total_img))
	images_list = []
	for i in range(global_total_img):
		img = mpimg.imread('data_provided/unzip_images/images/' + str(i + 1) + '.jpg')
		matrix[:,i] = np.shape(img)
		images_list.append(img)

	return images_list

# Test the accuracy of our model
def accuracy(model, arr_test, arr_labels):

	prediction = model.predict(arr_test)
	prediction = np.array(np.argmax(prediction, axis = 1))
	accur = np.array([1 for (a,b) in zip(prediction, arr_labels) if a==b ]).sum() / (global_num_test + 0.0)

	return accur, prediction

# Plot model statistics for keras models
# From https://www.kaggle.com/limitpointinf0/leaf-classifier-knn-nn?scriptVersionId=3352828
def plt_perf(name, p_loss=False, p_acc=False, val=False, size=(15,9), save=False):
	if p_loss or p_acc:
		if p_loss:
			plt.figure(figsize = size)
			plt.title('Loss')
			plt.plot(name.history['loss'], 'b', label='loss')
			if val:
				plt.plot(name.history['val_loss'], 'r', label='val_loss')
			plt.xlabel('Epochs')
			plt.ylabel('Value')
			plt.legend()
			plt.show()
			if save:
				plt.savefig('loss.png')
		if p_acc:
			plt.figure(figsize = size)
			plt.title('Accuracy')
			plt.plot(name.history['acc'], 'b', label='acc')
			if val:
 				plt.plot(name.history['val_acc'], 'r', label='val_acc')
			plt.xlabel('Epochs')
 			plt.ylabel('Value')
			plt.legend()
 			plt.show()
			if save:
 				plt.savefig('acc.png')
	else:
		print('No plotting since all parameters set to false.')

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
y_train = np_utils.to_categorical(y)

# Extract the id of each leaf
train_ids = train.pop('id')

# Set up our testing data
test = pd.read_csv('data_provided/test.csv')

# Extract the id of each leaf
test_ids = test.pop('id')

x_train = StandardScaler().fit(train).transform(train)
x_test = StandardScaler().fit(test).transform(test)

# Load up all of the images
img_list = grab_images()
# We will want to learn features on the images that belong to our training set
train_list = []

for j in range(len(train_ids)):
	for i in range(global_total_img):
		i = int(train_ids[j: j+1]) - 1
		if (int(train_ids[j: j+1]) == i + 1):
			train_list.append(img_list[i])
			break

# Visualize a leaf or two
visualize(train_list)

# Setting up the Keras neural network
# Create our model (currently has 2 hidden layers and using softmax regression)
model = Sequential()

model.add(Dense(global_hidden_layer1, input_dim = global_input_layer, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(global_hidden_layer2, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(global_output_layer, activation = 'softmax'))

# Compile our model
model.compile(optimizer = 'SGD', loss='categorical_crossentropy', metrics=['accuracy'])
print model.summary()

'''
Fit our model
Early stopping helps prevent over fitting but stopping out fitting function 
if our val_loss doesn't decrease after a certain number of epochs (called patience)
'''
early_stopper= EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
history = model.fit(x_train, y_train, epochs= global_epochs, batch_size = global_batch_size, verbose=1,
    validation_split=0.1, shuffle=True, callbacks=[early_stopper])

'''
INCORRECT CODE
# Evaluate our model
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
'''

# Check Keras' statistics
plt_perf(history, p_loss=True, p_acc=True, val=True)

# Test our model on the test set
y_pred = model.predict_proba(x_test)
print '\n'

'''
Let's see how accurace our model is
WARNING
This is based solely on low probability for guess = incorrect
Has been tested and shows correct guesses majority of the time (bad)
'''
x = []
x_val = []
total = 0
for i in range(len(y_pred)):
	if (np.amax(y_pred[i]) < 0.5):
		x = np.append(x, np.argmax(y_pred[i]))
		x_val = np.append(x_val, test_ids[i])
		total += 1
print 'Total number of < 0.5 predictions: %g' %(total)

x2 = np.ones(global_num_classes)
for i in range (len(x2)):
	x2[i] = i
for i in range(len(x)):
	for j in range(len(x2)):
		if (x2[j] == x[i]):	
			print classes[j]
			print int(x_val[i])
			img = mpimg.imread('data_provided/unzip_images/images/' + str(int(x_val[i])) + '.jpg')
			plt.title(classes[j])
			plt.imshow(img, cmap = 'binary')
			# plt.show()

# Set up the predictions into the correct format to submit to Kaggle
y_pred = pd.DataFrame(y_pred,index=test_ids,columns=classes)
'''
fp = open('submissionTest2.csv','w')
fp.write(y_pred.to_csv())
'''
print 'Finished.'

