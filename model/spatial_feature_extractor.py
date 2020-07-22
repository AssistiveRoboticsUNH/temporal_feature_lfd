import torch
import torch.nn as nn

class SpatialFeatureExtractor(nn.Module):   
	def __init__(self, 
			num_classes, 
			use_aud, 
			is_training, 
			checkpoint_file, 
			bottleneck_size,
		):

		super().__init__()

		self.num_classes = num_classes
		self.use_aud = use_aud

		# rgb net
		from backbone_model.tsm import TSM as VisualFeatureExtractor
		self.rgb_net = VisualFeatureExtractor(
			self.checkpoint_file, 
			self.num_classes, 
			training=is_training, 
			bottleneck_size=bottleneck_size)

		self.linear_dimension = self.rgb_net.size()

		# audio net
		if (self.use_aud):
			from aud.abc import AudioNetwork as AudioNetwork
			self.aud_net = AudioNetwork().getLogits()

			self.linear_dimension += self.aud_net.size()

		# pass to LSTM
		self.linear = Sequential(
			Linear(self.linear_dimension, self.num_classes)
		)

	# Defining the forward pass    
	def forward(self, rgb_x, aud_x):

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x)

		# if using audio data as well I need to combine those features
		if (self.use_aud):
			aud_y = self.aud_net(aud_x)
			obs_x = torch.stack([rgb_y, aud_y], dim=0, out=None)
		else:
			obs_x = rgb_y

		# pass through linear layer
		obs_y = self.linear(obs_x)

		return obs_y