import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor

class TemporalFeatureExtractor(FeatureExtractor): 
	def __init__(self, lfd_params, is_training):
		super().__init__(lfd_params, is_training)

		from .ditrl import DITRLWrapper
		filename = self.lfd_params.generate_ext_modelname()
		self.ditrl = DITRLWrapper(self.bottleneck_size, self.num_classes, is_training, filename)

	# Defining the forward pass    
	def forward(self, rgb_x):

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x) 
		
		# apply linear layer and consensus module to the output of the CNN
		#if (self.is_training):
		rgb_y = rgb_y.view((-1, self.rgb_net.num_segments) + rgb_y.size()[1:])
		#else:
		#	rgb_y = rgb_y.view((-1, self.rgb_net.num_segments*10) + rgb_y.size()[1:])

		# pass into D-ITR-L
		# ---
		return  self.ditrl(rgb_y)
		
	def save_model(self, debug=False):
		super().save_model(debug)

		self.ditrl.save_model(debug)

		
		