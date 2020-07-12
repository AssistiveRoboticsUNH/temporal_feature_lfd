import torch
import torch.nn as nn

class Net(nn.Module):
	def __init__(self, use_ditrl):
		super(Net, self).__init__(self)

		# Observation feature extractor
		# --------
		if(use_ditrl):
			from spatial_feature_extractor import SpatialFeatureExtractor
			self.observation_extractor = self.SpatialFeatureExtractor()
		else:
			from temporal_feature_extractor import TemporalFeatureExtractor
			self.observation_extractor = self.TemporalFeatureExtractor()

		# Policy Generator
		# --------
		self.policy_output = Sequential(
			Linear(4 * 7 * 7, 10)
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

	net = Net(use_ditrl = False)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	data, label = np.random.randint(256, size=(127, 127)), 0

	print(net(data).shape)



