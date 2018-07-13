# Import the necessary packages
import numpy as np
import matplotlib
matplotlib.use('Agg')                            # For running on the supercomputer
import matplotlib.pyplot as plt

####### Definitions #######
# Look at what some of the leaves look like
def visualize_leaves(images, y, show1, show2):
	'''
	#center_y, center_x = ndi.center_of_mass(img)
	print images[0,:,:,0].shape
	allm = np.concatenate((images[0,:,:,0], images[1,:,:,0], images[2,:,:,0], images[3,:,:,0]))
	plt.imshow(allm, cmap = 'binary')
	#plt.scatter(center_x, center_y)
	plt.show()
	'''
	if show1:
		# Setting up our grid
		num_row = 3
		num_col = 3
		for i in range(num_row):
			for j in range(num_col):
				if (j == 0):
					images1 = images[j + i * num_col,:,:,0]
		
				else:
					temp = images[j + i * num_col,:,:,0]
					images1 = np.concatenate((images1, temp), axis = 1)
				
			if (i == 0):
				images2 = images1
			else:
				images2 = np.concatenate((images2, images1), axis = 0)

		# Displaying the grid
		a = plt.figure(1)
		plt.imshow(images2, cmap = 'binary', interpolation = 'none')
		a.show()

	if show2:
		print y[0:20]
		count = 0
		for j in range(len(y)):
			if y[j] == 73:
				print j
				count += 1
				if count < 6:
					if (count == 1):
						pics = images[j,:,:,0]
		
					else:
						temp = images[j,:,:,0]
						pics = np.concatenate((pics, temp), axis = 1)
				else:
					if (count == 6):
						pics2 = images[j,:,:,0]
	
					else:
						temp = images[j,:,:,0]
						pics2 = np.concatenate((pics2, temp), axis = 1)
	
		al = np.concatenate((pics, pics2), axis = 0)

		b = plt.figure(2)
		plt.imshow(al, cmap = 'binary', interpolation = 'none')
		ax = plt.gca()
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		b.show()
	if show1 or show2:
		raw_input()
	return

# Plot model statistics for keras models
# From https://www.kaggle.com/limitpointinf0/leaf-classifier-knn-nn?scriptVersionId=3352828
def plt_perf(name, f1, f2, p_loss=False, p_acc=False, val=False, size=(15,9), save=False):
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
			# plt.show()
			if save:
				print 'Saving loss...'
				plt.savefig(f1)
		if p_acc:
			plt.figure(figsize = size)
			plt.title('Accuracy')
			plt.plot(name.history['acc'], 'b', label='acc')
			if val:
 				plt.plot(name.history['val_acc'], 'r', label='val_acc')
			plt.xlabel('Epochs')
 			plt.ylabel('Value')
			plt.legend()
 			# plt.show()
			if save:
				print 'Saving accuracy...'
 				plt.savefig(f2)
	else:
		print('No plotting since all parameters set to false.')

	return
