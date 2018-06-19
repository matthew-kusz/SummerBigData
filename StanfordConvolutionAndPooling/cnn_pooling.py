# Import the necessary packages
import numpy as np

def pooling(pool_dim, convolve_feat):
	'''
	Parameters:
	pool_dim - dimension of pooling region (19)
	convolve_feat - convolved features to pool (400, m, 57, 57)

	Returns:
	pooled_features - matrix of pooled features (400, m, 3, 3)
	'''

	# Number of convolve_feats
	m = convolve_feat.shape[1]
	# Dimension of convolve_feat
	convolved_dim = convolve_feat.shape[2]
	# Number of channels
	num_features = convolve_feat.shape[0]

	pooled_features = np.zeros((num_features, m, np.floor(convolved_dim / pool_dim), np.floor(convolved_dim / pool_dim))) # (400, m, 3, 3)

	# We need to divide our convolved features into 9 different sections and find the mean feature activation over these regions
	# For each row
	for j in range(convolved_dim / pool_dim):
		# For each column
		for k in range(convolved_dim / pool_dim):
			temp = np.zeros((num_features, m, pool_dim, pool_dim))
			temp = convolve_feat[:, :, j * pool_dim: pool_dim + j * pool_dim, k * pool_dim: pool_dim + k * pool_dim]
			pooled_features[:, : ,j, k] = np.mean(temp, axis = (2, 3))

	return pooled_features
