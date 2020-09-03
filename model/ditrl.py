import numpy as np 
import scipy

import torch
import torch.nn as nn

import subprocess, os, tempfile
from scipy.signal import savgol_filter
from .parser_utils import write_sparse_matrix, read_itr_file

from sklearn.linear_model import SGDClassifier

class DITRLWrapper(nn.Module):
	def __init__(self, num_features, num_classes, is_training, modelname):
		super().__init__()

		self.ditrl = DITRLPipeline(num_features, is_training)
		self.model = DITRL_Linear(num_features, num_classes, is_training, modelname)

	def forward(self, activation_map):
		activation_map = activation_map.detach().cpu().numpy()

		iad 		= self.ditrl.convert_activation_map_to_IAD(activation_map)
		sparse_map  = self.ditrl.convert_IAD_to_sparse_map(iad)
		itr 		= self.ditrl.convert_sparse_map_to_ITR(sparse_map)

		# pre-process ITRS
		# scale / TFIDF

		# evaluate on ITR
		itr = itr.astype(np.float32)
		itr = torch.autograd.Variable(torch.from_numpy(itr).cuda())
		return self.model(itr)

	def save_model(self, debug=False):
		self.model.save_model(debug)

class DITRLPipeline: # pipeline
	def __init__(self, num_features, is_training):

		

		self.is_training = is_training

		self.num_features = num_features
		self.threshold_values = np.zeros(self.num_features, np.float32)
		self.threshold_file_count = 0

		self.scaler = None
		self.TFIDF = None

	# ---
	# extract ITRs
	# ---

	def convert_activation_map_to_IAD(self, activation_map, save_name="", ):
		# reshape activation map
		# ---

		iad = np.reshape(activation_map, (-1, self.num_features))
		iad = iad.T

		# pre-processing of IAD
		# ---

		# trim start noisy start and end of IAD
		if(iad.shape[1] > 10):
			iad = iad[:, 3:-3]

		# use savgol filter to smooth the IAD
		smooth_window = 35
		if(iad.shape[1] > smooth_window):
			for i in range(iad.shape[0]):
				iad[i] = savgol_filter(iad[i], smooth_window, 3)

		# update threshold
		# ---
		if (self.is_training):

			altered_value = self.threshold_values * self.threshold_file_count
			self.threshold_values += np.mean( iad , axis=1)
			self.threshold_file_count += 1

			self.threshold_values /= self.threshold_file_count

		# return IAD
		# ---
		return iad

	def convert_IAD_to_sparse_map(self, iad):
		'''Convert the IAD to a sparse map that denotes the start and stop times of each feature'''

		# apply threshold
		# ---

		# threshold, reverse the locations to account for the transpose
		locs = np.where(iad > self.threshold_values.reshape(self.num_features, 1))
		locs = np.dstack((locs[1], locs[0]))

		# get the start and stop times for each feature in the IAD
		if(len(locs) != 0):
			sparse_map = []
			for i in range(iad.shape[0]):
				feature_row = locs[np.where(locs[:,0] == i)][:,1]

				# locate the start and stop times for the row of features
				start_stop_times = []
				if(len(feature_row) != 0):
					start = feature_row[0]
					for i in range(1, len(feature_row)):

						if( feature_row[i-1]+1 < feature_row[i] ):
							start_stop_times.append([start, feature_row[i-1]+1])
							start = feature_row[i]

					start_stop_times.append([start, feature_row[len(feature_row)-1]+1])

				# add start and stop times to sparse_map
				sparse_map.append( start_stop_times )
		else:
			sparse_map = [[] for x in xrange(iad.shape[0])]

		# ---
		return sparse_map

	def convert_sparse_map_to_ITR(self, sparse_map, cleanup=True):

		# create files
		file_id = next(tempfile._get_candidate_names())
		sparse_map_filename = os.path.join("/tmp",file_id+".b1")
		itr_filename = os.path.join("/tmp",file_id+".b2")

		# write the sparse map to a file
		write_sparse_matrix(sparse_map_filename, sparse_map)

		# execute the itr identifier (C++ code)
		subprocess.call(["model/itr_parser", sparse_map_filename, itr_filename])

		#open ITR file
		itrs = read_itr_file(itr_filename)

		#file cleanup
		if (cleanup):
			os.system("rm "+sparse_map_filename+" "+itr_filename)

		return itrs


'''
class DITRL_SVM:
	def __init__(self, num_features, num_classes, is_training):
		alpha = 0.001
		n_jobs = 4
		self.model = SGDClassifier(loss='hinge', alpha=alpha, n_jobs=n_jobs)

		self.num_classes = num_classes

	def forward(self, data):
		data = scipy.sparse.coo_matrix(data)
		return self.model.predict(data)

	def train(self, data, label):
		data = scipy.sparse.coo_matrix(data)
		label = np.array(label)
		net.partial_fit(data, label, classes=np.arange(self.num_classes))
'''
class DITRL_Linear(nn.Module):
	def __init__(self, num_features, num_classes, is_training, modelname):
		super().__init__()

		self.inp_dim = num_features * num_features * 7
		self.num_classes = num_classes
		self.modelname = modelname

		self.model = nn.Sequential(
			nn.Linear(self.inp_dim, self.num_classes)
		)


		# load a previously saved model
		if (not is_training):
			ext_checkpoint = self.modelname
			if (ext_checkpoint):

				# load saved model parameters	
				print("ditrl.py: Loading Extension Model from: ", ext_checkpoint)	
				checkpoint = torch.load(ext_checkpoint)


				print("var:")
				for k, v in checkpoint.items():
					print("k", k)
				print("var done")

				self.model.load_state_dict(checkpoint, strict=True)

				# prevent changes to these parameters
				for param in self.model.parameters():
					param.requires_grad = False	
			else:
				print("ditrl.py: Did Not Load Extension Model")	

	def forward(self, data):
		data = torch.reshape(data, (-1, self.inp_dim))
		return self.model(data)

	def save_model(self, debug=False):
		if (debug):
			print("ditrl.state_dict():")
			for k in self.model.state_dict().keys():
				print("\t"+k, self.model.state_dict()[k].shape )

		torch.save(self.model.state_dict(), self.modelname )
		print("Ext model saved to: ", self.modelname)