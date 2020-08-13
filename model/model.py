import torch
import torch.nn as nn

class LfDNetwork(nn.Module):
	def __init__(self, lfd_params, is_training=False):
		super().__init__()

		# Observation feature extractor
		# --------
		
		if(lfd_params.args.use_ditrl):
			from .temporal_feature_extractor import TemporalFeatureExtractor
			self.observation_extractor = TemporalFeatureExtractor()
		else:
			from .spatial_feature_extractor import SpatialFeatureExtractor
			self.observation_extractor = SpatialFeatureExtractor(
				lfd_params.num_actions, 
				lfd_params.use_aud, 
				is_training, 
				lfd_params.checkpoint_file, 
				)
			
		# Policy Generator
		# --------
		self.policy_output = nn.Sequential(
			nn.Linear(lfd_params.num_actions + 1, lfd_params.num_actions)
		)

	# Defining the forward pass    
	def forward(self, obs_x, hidden_x):

		#extract visual features from observation
		obs_y = self.observation_extractor(obs_x)

		#combine visual features with hidden world state
		state_x = torch.stack([obs_y, hidden_x], dim=0, out=None)

		#obtain logits
		state_y = self.linear_layers(state_x)

		return state_y

	def select_action(self, obs_x, hidden_x):

		#choose action with highest value
		state_y = self.forward(obs_x, hidden_x)
		action = np.argmax(state_y)
		return action

if __name__ == '__main__':
	import numpy as np

	net = LfDNetwork(use_ditrl = False)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	data, label = np.random.randint(256, size=(127, 127)), 0

	print(net(data).shape)



