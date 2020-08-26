import torch
import torch.nn as nn

from .spatial_feature_extractor import SpatialFeatureExtractor

class TemporalFeatureExtractor(SpatialFeatureExtractor): 
	def __init__(self, lfd_params, is_training):

		super().__init__(lfd_params, is_training)

		from .ditrl import DITRLWrapper
		self.ditrl = DITRLWrapper(self.bottleneck_size)
		
		self.linear_dimension = self.ditrl.output_size
		
		# pass to LSTM
		self.linear = nn.Sequential(
			nn.Linear(self.linear_dimension, self.num_classes)
		)

	# Defining the forward pass    
	def forward(self, rgb_x):

		# extract IAD
		# ---

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x) 
		
		# apply linear layer and consensus module to the output of the CNN
		if (self.is_training):
			rgb_y = rgb_y.view((-1, self.rgb_net.num_segments) + rgb_y.size()[1:])
		else:
			rgb_y = rgb_y.view((-1, self.rgb_net.num_segments*10) + rgb_y.size()[1:])
		#rgb_y = self.consensus(rgb_y)
		#rgb_y = rgb_y.squeeze(1)

		# pass into D-ITR-L
		# ---
		rgb_y = self.ditrl(rgb_y)




		obs_y = self.linear(rgb_y)

		return obs_y

		# if using audio data as well I need to combine those features
		'''
		if (self.use_aud):
			aud_y = self.aud_net(aud_x)
			obs_x = torch.stack([rgb_y, aud_y], dim=0, out=None)
		else:
			obs_x = rgb_y
		'''


