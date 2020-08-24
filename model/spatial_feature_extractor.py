import torch
import torch.nn as nn

from .backbone_model.tsm.ops.basic_ops import ConsensusModule

class SpatialFeatureExtractor(nn.Module):   
	def __init__(self, 
			num_classes, 
			use_aud, 
			is_training, 
			checkpoint_file, 
		):

		super().__init__()

		self.num_classes = num_classes
		self.use_aud = use_aud
		self.is_training = is_training
		self.checkpoint_file = checkpoint_file

		self.bottleneck_size = 128

		# rgb net
		from .backbone_model.tsm.tsm import TSMWrapper as VisualFeatureExtractor
		self.rgb_net = VisualFeatureExtractor(
			self.checkpoint_file, 
			self.num_classes, 
			training=is_training)

		self.maxpool = nn.Sequential(
			nn.Conv2d(2048, self.bottleneck_size, (1,1)),
			nn.AdaptiveMaxPool2d(output_size=1),
		)
		'''
		self.maxpool = nn.Sequential(
			nn.Conv2d(2048, self.bottleneck_size, (1,1)),
			nn.AdaptiveMaxPool2d(output_size=1),
		)
		'''

		self.linear_dimension = self.rgb_net.bottleneck_size
		'''
		# audio net
		if (self.use_aud):
			from aud.abc import AudioNetwork as AudioNetwork
			self.aud_net = AudioNetwork().getLogits()

			self.linear_dimension += self.aud_net.size()
		'''
		# pass to LSTM
		self.linear = nn.Sequential(
			nn.Linear(self.linear_dimension, self.num_classes)
		)

		self.consensus = ConsensusModule('avg')

	# Defining the forward pass    
	def forward(self, rgb_x):

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x)
		rgb_y = self.maxpool(rgb_y)
		print("rgb_y size:", rgb_y.size())

		# if using audio data as well I need to combine those features
		'''
		if (self.use_aud):
			aud_y = self.aud_net(aud_x)
			obs_x = torch.stack([rgb_y, aud_y], dim=0, out=None)
		else:
			obs_x = rgb_y
		'''
		obs_x = rgb_y

		# pass through linear layer
		obs_y = self.linear(obs_x)
		obs_y = self.consensus(obs_y)

		return obs_y