# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt
import data_setup
from scipy.ndimage import rotate
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics.pairwise import cosine_similarity   # Used for checking how similar images are to one another

global_max_dim = 50

seed = 2
np.random.seed(seed)

def augment(train, train_mod_list, y, y_train, test_mod_list, test):
	# Let's create more images for us
	datagen = ImageDataGenerator(rotation_range=20, fill_mode='nearest')

	datagen.fit(train_mod_list)

	# Configure batch size and retrieve one batch of images
	temp = []
	temp2 = []
	temp3 = []
	temp4 = []
	temp5 = []

	te = []
	te2 = []
	for i in range(len(train_mod_list)):
		j = 0
		temp.append(train_mod_list[i].ravel())
		temp2.append(y_train[i])
		temp3.append(train[i:i + 1])
		temp5.append(y[i])		
		
		
		for X_batch, y_batch in datagen.flow(train_mod_list[i:i + 1], y_train[i:i + 1], batch_size=9):
			temp.append(X_batch.ravel())
			temp2.append(y_batch)
			temp3.append(train[i:i + 1])
			temp5.append(y[i])
		
			j += 1
			if j == 9:
				break

	for i in range(len(test_mod_list)):
		j = 0
		te.append(test_mod_list[i].ravel())
		te2.append(test[i:i+1])
		for Z_batch in datagen.flow(test_mod_list[i:i + 1], batch_size=9):
			te.append(Z_batch.ravel())
			te2.append(test[i:i + 1])

			j += 1
			if j == 9:
				break		

	
	temp_arr = np.zeros((len(temp), global_max_dim * global_max_dim))
	temp2_arr = np.zeros((len(temp2), 99))
	temp3_arr = np.zeros((len(temp3), 222))
	temp5_arr = np.zeros((len(temp5), 1))
	print temp5_arr.shape

	for i in range(len(temp)):
		temp_arr[i] = temp[i]
		temp2_arr[i] = temp2[i]
		temp3_arr[i] = temp3[i]
		temp5_arr[i] = temp5[i]
	
	all_all = np.concatenate((temp_arr, temp2_arr, temp3_arr, temp5_arr), axis = 1)
	np.random.shuffle(all_all)
	
	for i in range(len(temp)):
		temp_arr[i] = all_all[i][0: temp_arr.shape[1]]
		temp2_arr[i] = all_all[i][temp_arr.shape[1]: temp_arr.shape[1] + 99]
		temp3_arr[i] = all_all[i][temp_arr.shape[1] + 99: temp_arr.shape[1] + 99 + 222]
		temp5_arr[i] = all_all[i][temp_arr.shape[1] + 99 + 222: temp_arr.shape[1] + 99 + 222 + 1]
	
	train_mod_list = temp_arr.reshape(len(temp_arr), global_max_dim, global_max_dim, 1)
	y_train = temp2_arr
	train = temp3_arr
	train_list = temp4
	y = temp5_arr.ravel()

	te_arr = np.zeros((len(te), global_max_dim * global_max_dim))
	te2_arr = np.zeros((len(te2), 222))
	for i in range(len(te)):
		te_arr[i] = te[i]
		te2_arr[i] =te2[i]

	return train, y , y_train, train_mod_list, te_arr.reshape(len(te_arr), global_max_dim, global_max_dim, 1), te2_arr

