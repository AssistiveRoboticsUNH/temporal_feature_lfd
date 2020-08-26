import numpy as np 
from scipy.signal import savgol_filter

# plan to always use the activation map and work back from there

class DITRL:
	def __init__(self, use_generated_files):
		self.output_file = None
		self.use_generated_files = use_generated_files

		self.threshold_values = np.zeros(np.float32, self.num_features)
		self.threshold_file_count = 0

		self.scaler = None
		self.TFIDF = None

		self.training = True

	def forward(self, activation_map):
		# extract ITRs
		iad 		= self.convert_activation_map_to_IAD(activation_map)
		sparse_map  = self.convert_IAD_to_sparse_map(iad)
		itr 		= self.convert_sparse_map_to_ITR(sparse_map)

		# pre-process ITRS
		# scale / TFIDF

		# evaluate on ITR

	# ---
	# extract ITRs
	# ---

	def convert_activation_map_to_IAD(self, activation_map, save_name="", ):
		# reshape activation map
		# ---

		# perform max? 
		# ---

		# pre-processing of IAD
		# ---

		# trim start noisy start and end of IAD
		if(iad.shape[1] > 10):
			iad = iad[:, 3:-3]

		# use savgol filter to smooth the IAD
		smooth_value = 25
		if(layer >= 1):
			smooth_value = 35
		
		if(iad.shape[1] > smooth_value):
			for i in range(iad.shape[0]):
				iad[i] = savgol_filter(iad[i], smooth_value, 3)

		# update threshold
		# ---
		if (self.training):

			altered_value = self.threshold_values * self.threshold_file_count
			self.threshold_values += np.mean( iad , axis=1)
			self.threshold_file_count += 1

			self.threshold_values /= self.threshold_file_count

		# save as IAD as npy file
		# ---



		# return IAD
		# ---
		return iad

	def convert_IAD_to_sparse_map(self, iad):
		'''Convert the IAD to a sparse map that denotes the start and stop times of each feature'''

		# apply threshold
		# ---

		# threshold, reverse the locations to account for the transpose

		locs = np.where(iad.T > threshold_values)
		locs = np.array( zip( locs[1], locs[0] ) )

		# get the start and stop times for each feature in the IAD
		if(len(locs) != 0):
			sparse_map = []
			for i in range(iad.shape[0]):
				feature_times = locs[np.where(locs[:,0] == i)][:,1]
				sparse_map.append( find_start_stop( feature_times ))
		else:
			sparse_map = [[] for x in xrange(iad.shape[0])]

		# write start_stop_times to file.
		# ---
		
		#write_sparse_matrix(ex['b_path'], sparse_map)

		# return sparse_map
		# ---
		return sparse_map

	def convert_sparse_map_to_ITR(self, sparse_map):
		# execute c++ code

		# return iad