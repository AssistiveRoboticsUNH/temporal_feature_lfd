import torch
import torch.nn as nn

class TemporalFeatureExtractor(nn.Module):   
	def __init__(self, use_aud):
		super().__init__()

	def temporal_feature_extractor():
		# rgb net
		from xyz.abc import DITRL
		self.rgb_net = DITRL()

		# audio net
		from aud.abc import AudioNetwork as AudioNetwork
		self.aud_net = AudioNetwork().getLogits()

		# combine the values together using D-ITR-L

		return None

	# Defining the forward pass    
	def forward(self, rgb_x, aud_x):
		obs_y = self.observation_extractor(obs_x)

		state_x = torch.stack([obs_y, hidden_x], dim=0, out=None)

		state_y = self.linear_layers(state_x)

		return state_y