import torch
import torch.nn as nn

class SpatialFeatureExtractor(nn.Module):   
	def __init__(self):
		super(SpatialFeatureExtractor, self).__init__()

		# rgb net
		from xyz.abc import TSMBackBone as VisualFeatureExtractor
		self.rgb_net = VisualFeatureExtractor().getLogits()

		# audio net
		from aud.abc import AudioNetwork as AudioNetwork
		self.aud_net = AudioNetwork().getLogits()

		# pass to LSTM
		self.linear = Sequential(
			Linear(4 * 7 * 7, 10)
		)

	# Defining the forward pass    
	def forward(self, rgb_x, aud_x):

		# pass data through CNNs
		rgb_y = self.rgb_net(rgb_x)
		aud_y = self.aud_net(aud_x)

		# merge data together
		obs_x = torch.stack([rgb_y, aud_y], dim=0, out=None)

		# pass through LSTM
		obs_y = self.linear(obs_x)

		return obs_y