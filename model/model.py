import torch
import torch.nn as nn
import numpy as np

class LfDNetwork(nn.Module):
	def __init__(self, lfd_params, is_training=False):
		super().__init__()

		self.use_ditrl = lfd_params.args.use_ditrl
		self.trim_model = lfd_params.args.trim_model

		# Observation feature extractor
		# --------
		
		if(self.use_ditrl):
			from .temporal_feature_extractor import TemporalFeatureExtractor as FeatureExtractor
		else:
			from .spatial_feature_extractor import SpatialFeatureExtractor as FeatureExtractor
			
		self.observation_extractor = FeatureExtractor( lfd_params, is_training )
			
		# Policy Generator
		# --------
		self.policy_output = nn.Sequential(
			nn.Linear(lfd_params.num_actions + lfd_params.num_hidden_state_params, lfd_params.num_actions)
		)

	# Defining the forward pass    
	def forward(self, obs_x, state_x):

		#extract visual features from observation
		obs_y = self.observation_extractor(obs_x)

		if (self.trim_model):
			return obs_y

		#combine visual features with hidden world state
		state_x = state_x.type(torch.FloatTensor).view([-1, 1]).cuda()
		state_x = torch.cat([obs_y, state_x], dim=1, out=None)

		#obtain logits
		state_y = self.policy_output(state_x)

		return state_y


if __name__ == '__main__':
	import numpy as np

	net = LfDNetwork(use_ditrl = False)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	data, label = np.random.randint(256, size=(127, 127)), 0

	print(net(data).shape)



