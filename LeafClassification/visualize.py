# Import the necessary packages
import numpy as np
import matplotlib.pyplot as plt                          # For plotting
import matplotlib.image as mpimg           	 	 # Reading images to numpy arrays
import matplotlib.cm as cm                               # For matplotlib's cmap
from sklearn.metrics.pairwise import cosine_similarity   # Used for checking how similar images are to one another
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

####### Definitions #######
def visualize_leaves(images, y, classes, show1, show2):
	'''
	Create a grid of random leaves and/or display all the images of a specified leaf species
	Displaying the specifed species works for displaying guesses of the test set too

	Parameters:
	images - list of images of the leaves
	y - list of numbers ranging from 0-98 where each number represents a species of leaf
	classes - list of the names of the species
	show1 - shows the grid of leaves if true
	show2 - shows the images of a specified species of leaves if true	
	'''

	# For displaying a grid of random leaves
	if show1:
		# Setting up our grid
		num_row = 15
		num_col = 15

		# Setting up each row
		for i in range(num_row):
			# Setting up each image in said row
			for j in range(num_col):
				if (j == 0):
					images1 = images[j + i * num_col,:,:,0]
		
				else:
					temp = images[j + i * num_col,:,:,0]
					images1 = np.concatenate((images1, temp), axis = 1)
			
			# Attach new row below previous one	
			if (i == 0):
				images2 = images1
			else:
				images2 = np.concatenate((images2, images1), axis = 0)

		# Displaying the grid
		a = plt.figure(1)
		plt.imshow(images2, cmap = 'binary', interpolation = 'none')
		a.show()

	# For displaying images of a specified leaf species (Works for guessed leaves in test set)
	if show2:
		
		# Give a list of the id associated with each species
		for i in range(len(classes)):
			print i, classes[i]
		
		# Ask for what species to display	 
		species = input('What leaf species would you like to look at? ')
		print 'You chose', classes[species]

		# Find how many images of that species there is
		hold = []
		num_img = 0

		# Create a list holding each index that image is in
		for i in range(len(y)):
			if y[i] == species:
				num_img += 1
				hold.append(i)

		'''
		Check if we have an odd amount of images
		(relevant for displaying the test leaves after predictions)
		'''

		if num_img % 2 != 0:
			print 'Number of images is not even!'
			# Print the odd image separately
			c = plt.figure(1)
			plt.imshow(images[hold[num_img - 1],:,:,0], cmap = 'binary', interpolation = 'none')
			ax = plt.gca()
			ax.axes.get_xaxis().set_visible(False)
			ax.axes.get_yaxis().set_visible(False)
			c.show()
			num_img -= 1	
		
		# Display the images in 2 rows
		count = 0
		for j in range(len(y)):
			if y[j] == species:
				count += 1
				# First row
				if count < (num_img / 2) + 1:
					if (count == 1):
						pics = images[j,:,:,0]
		
					else:
						temp = images[j,:,:,0]
						pics = np.concatenate((pics, temp), axis = 1)
				# Second row
				elif count > (num_img / 2) and count < num_img + 1:
					if (count == (num_img / 2) + 1):
						pics2 = images[j,:,:,0]
	
					else:
						temp = images[j,:,:,0]
						pics2 = np.concatenate((pics2, temp), axis = 1)
	
		# Display the images
		al = np.concatenate((pics, pics2), axis = 0)

		b = plt.figure(num_img)
		plt.imshow(al, cmap = 'binary', interpolation = 'none')
		ax = plt.gca()
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		b.show()

	# Show multiple images at once if need be
	if show1 or show2:
		raw_input()
	return

def image_similarity(train, y, classes):
	'''
	See how similar each image is to one another of the same leaf species
	
	Parameters:
	train - list of images in the training set
	y - list of numbers ranging from 0-98 where each number represents a species of leaf
	classes - list of the names of the species
	'''
	
	# Cycle through each class
	for i in range(len(classes)):
		# Find all of the image of said class
		temp = np.where(y == i)

		# Stick all of the images in a list
		temp1 = []
		for j in range(len(temp[0])):
			temp1.append(train[temp[0][j],:,:,0])

		# Use the first image of the list to compare to the remaining 9
		summ = 0	
		for k in range(len(temp1) - 1):
			summ += np.asscalar(cosine_similarity(temp1[0].reshape(1,-1), temp1[k + 1].reshape(1,-1)))

		# Print the average cosine similarity for each species
		print classes[i], 'cosine similarity:', summ / (len(temp1) - 1)

	return

def plt_perf(name, f1, f2, p_loss=False, p_acc=False, val=False, size=(15,9), save=False):
	'''
	From https://www.kaggle.com/limitpointinf0/leaf-classifier-knn-nn?scriptVersionId=3352828
	Plot the model stats for the keras model (accuracy, loss)
	
	Parameters:
	name - model used
	f1 - filename to save loss plot to
	f2 - filename to save accuracy plot to
	p_loss - tells if we want to display the loss plot
	p_acc - tells if we want ot display the accuracy plot
	val- tells if we want to show the stats of our validation set
	size - size of our plots
	save - tells if we want to save our plots or not
	'''

	if p_loss or p_acc:
		# Display loss plot
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

		# Display accuracy plot
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
		print 'No plotting since all parameters set to false.'

	return

def visualize_PCA(tr_pca, tr_m_list, pca):
	'''
	Visualize the PCA images and the explained_variance_ratio

	Parameters:
	tr_pca - 2D array of the training set flattened pca images
	tr_m_list - list of the resized training images
	pca - pca function that was set up in data_setup
	'''

	print pca.explained_variance_ratio_
	print np.sum(pca.explained_variance_ratio_)
	approximation = pca.inverse_transform(tr_pca)

	# Relationship of information retained vs. number of prinicpal components
	mark = [15, 40, 70, 300]
	plt.plot(np.cumsum(pca.explained_variance_ratio_), color='b')
	plt.title('Information Retained vs. Principal Components')
	plt.ylabel('Fraction of Information Retained')
	plt.xlabel('Number of Principal Components')
	plt.show()

	plt.figure(figsize=(8,4))
	# Original Image
	plt.subplot(1, 2, 1)
	plt.imshow(tr_m_list[0].reshape(50,50),
              	cmap = plt.cm.gray, interpolation='nearest',
              	clim=(0, 1))
	ax = plt.gca()
	ax.axes.set_xticklabels([])
	ax.axes.get_yaxis().set_visible(False)
	plt.xlabel('2500 components', fontsize = 14)
	plt.title('Original Image', fontsize = 20)

	# number of principal components
	plt.subplot(1, 2, 2)
	plt.imshow(approximation[0].reshape(50, 50),
              	cmap = plt.cm.gray, interpolation='nearest',
              	clim=(0, 1))
	ax = plt.gca()
	ax.axes.set_xticklabels([])
	ax.axes.get_yaxis().set_visible(False)
	plt.xlabel(str(pca.n_components_) + ' components', fontsize = 14)
	plt.title('PCA Image', fontsize = 20)
	plt.show()

	return

def display_conf(y, top3, tr_img, tot):
	'''
	Displays images of the top 3 predicted species for the more confused species

	Parameters:
	y - 1D array that holds a label for each species ranging from 0 to n - 1
	top3 - number representing the species of leaf
	tr_img - list of the resized training images
	tot - keeps track of how many images there are

	Returns the images of the 3 species concatenated together
	'''
	# Display the images in 2 rows
	count = 0
	for j in range(len(y)):
		if y[j] == top3:
			count += 1
			# First row
			if count < 6:
				if (count == 1):
					pics = tr_img[j,:,:,0]
		
				else:
					temp = tr_img[j,:,:,0]
					pics = np.concatenate((pics, temp), axis = 1)
			# Second row
			else:
				if (count == 6):
					pics2 = tr_img[j,:,:,0]
	
				else:
					temp = tr_img[j,:,:,0]
					pics2 = np.concatenate((pics2, temp), axis = 1)
	
	# Display the images
	al = np.concatenate((pics, pics2), axis = 0)
	return al

def confusion(y_pred, y, classes, test_ids, num_classes, train_img, thresh):
	'''
	Checks what leaves had a low probability of being chosen and displays them, their guessed species,
	and the top three probabilities

	Parameters:
	y_pred - predictions for each leaf
	y - 1D array that holds a label for each species ranging from 0 to n - 1
	classes - 1D array of the name of each leaf species
	test_ids - 1D array of the testing set leaves ids
	num_classes - total number of classes
	train_img - list of the resized training images
	thresh - minimum probability needed to not be considered a confused leaf
	'''

	# Find the leaves that have lower probabilities 
	x = []
	x_val = []
	y_val = []
	total = 0
	for i in range(len(y_pred)):
		if (np.amax(y_pred[i]) < thresh):
			# Index where the highest probability is
			x = np.append(x, np.argmax(y_pred[i]))
			# ID of the leaf we are looking at
			x_val = np.append(x_val, test_ids[i])
			# Where the leaf is in the matrix (what row)
			y_val = np.append(y_val, i)
			# Tally up the total number of leaves
			total += 1
	print 'Total number of <', thresh, ' predictions: %g' %(total)

	# Set up an array that ranges from 0 - 98 to set up the class numbers
	x2 = np.ones(num_classes)
	for i in range(len(x2)):
		x2[i] = i

	most_conf = np.zeros(len(x2))
	# Cycle through each leaf that had a low probability
	for i in range(len(x)):
		total = 0
		# Cycle through each class number
		for j in range(len(x2)):
			# Display the leaves with the lower probabilities
			if (x2[j] == x[i]):
				# Find what the guess was
				print 'Species guessed:', classes[j]
				# Find the id of the corresponding image
				print 'ID associated with leaf:', int(x_val[i])
				
				# Get top 3 predictions
    				top3_ind = y_pred[int(y_val[i])].argsort()[-3:]
    				top3_species = np.array(classes)[top3_ind]
    				top3_preds = y_pred[int(y_val[i])][top3_ind]
				
				for m in range (len(top3_ind)):	
					most_conf[top3_ind[m] - 1] += 1 

    				# Display the top 3 predictions and the actual species
    				print("Top 3 Predicitons:")
    				for k in range(2, -1, -1):
        				print "\t%s (%s): %s" % (top3_species[k], top3_ind[k], top3_preds[k])

				# Display the image of the leaf
				string1 = 'data_provided/unzip_images/images/' + str(int(x_val[i])) + '.jpg'
				string2 = 'Species guessed: ' + classes[j] + ', ID: ' + str(int(x_val[i]))
				
				total += 1
				a = plt.figure(total)
				img = mpimg.imread(string1)
				plt.title(string2)
				ax = plt.gca()
				ax.axes.get_xaxis().set_visible(False)
				ax.axes.get_yaxis().set_visible(False)
				plt.imshow(img, cmap = 'binary')
				a.show()

				# Display the probabilities for that leaf
				total += 1
				b = plt.figure(total)
				plt.bar(x2, y_pred[int(y_val[i])])
				plt.title('Probability of Each Class for ID: ' + str(int(x_val[i])))
				plt.xlabel('Class Number')
				plt.ylabel('Probability')
				b.show()

				# Display the top three guesses
				images = np.zeros((3, 100, 250))
				for z in range(len(top3_ind)): 
					images[z] = display_conf(y, top3_ind[z], train_img, total)

				all_img = np.concatenate((images[2], images[1], images[0]), axis = 0)
				total += 1
				c = plt.figure(total)
				plt.imshow(all_img, cmap = 'binary')
				plt.title('Top 3 Guesses')
				ax = plt.gca()
				ax.axes.get_xaxis().set_visible(False)
				ax.axes.get_yaxis().set_visible(False)
				c.show()
				
				raw_input()

				plt.close('all')
				print '\n'


	print 'Classes confused'
	total = 0
	for i in range(len(x2)):
		if most_conf[i]	> 0:
			print classes[i] + '(' + str(i + 1) + ')' + ':' + str(most_conf[i])

	return

def visualize_tsne(images, ids, labels, fig_size = (20, 20)):
	# label_tp_id_dict = {v:i for i, v in enumerate(np.unique(labels))}
	# id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}

	plt.figure(figsize = fig_size)
	plt.grid()

	nb_classes = len(np.unique(ids))
	
	for label in np.unique(ids):
		plt.scatter(images[np.where(ids == label), 0], images[np.where(ids == label), 1],
				marker = 'o', color = plt.cm.Set1(label / float(nb_classes)),
				linewidth = '1', alpha = 0.8, label = labels[label])

	plt.legend(loc = 'best')
	return

def visualize_tsne_images(images, orig_images, fig_size = (50, 50), image_zoom = 0.6):
	orig_images = orig_images.reshape(len(orig_images),50, 50)
	fig, ax = plt.subplots(figsize=fig_size)
    	artists = []
	for xy, i in zip(images, orig_images):
        	x0, y0 = xy
        	img = OffsetImage(i, zoom=image_zoom, cmap = 'binary')
        	ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        	artists.append(ax.add_artist(ab))

   	ax.update_datalim(images)
    	ax.autoscale()
    	plt.show()
