# Import the necessary packages
import numpy as np

####### Global variables #######
global_step = 0
global_image_dim = 64
global_image_channels = 3
global_patch_dim = 8
global_num_patches = 50000
global_visible_size = global_patch_dim * global_patch_dim * global_image_channels
global_output_size = global_visible_size
global_hiddien_size = 400
global_epsilon = 0.1
global_pool_dim = 19

####### Definitions #######


####### Code #######
# First we need to grab the theta values, ZCA_matrix and mean patches we obtained from our linear decoder
final_theta = np.genfromtxt('/users/PAS1383/osu10173/work/GitHub/MattRepo/StanfordLearningColorFeaturesWithSparseAutoencoders/outputs/finalWeightsRho0.035Lambda0.003Beta5.0Size100000HL400.out')
ZCA_matrix = np.genfromtxt('/users/PAS1383/osu10173/work/GitHub/MattRepo/StanfordLearningColorFeaturesWithSparseAutoencoders/outputs/ZCAwhitening0.035Lambda0.003Beta5.0Size100000HL400.out')
mean_patches = np.genfromtxt('/users/PAS1383/osu10173/work/GitHub/MattRepo/StanfordLearningColorFeaturesWithSparseAutoencoders/outputs/meanPatchesRho0.035Lambda0.003Beta5.0Size100000HL400.out')

