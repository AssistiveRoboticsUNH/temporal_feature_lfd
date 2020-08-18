import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os, sys

#sys.path.insert(0, "/home/mbc2004/temporal-shift-module")
#from ops.transforms import * 
from model.backbone_model.tsm.ops.transforms import *

'''
File structure is:
  /root
	/train
	/validate
	/eval
	  /obs1
	  /obs2
	  /obs3
	  	/example1
	  	/example2
	  	/example3
			/00000.jpg frame1
			/00001.jpg frame2
			/00002.jpg frame3

Here is what I need to do: The dataset loader needs to open the video file and output 
the video, state history, and the action label. 

I need to get this working with an example () use Something SOmething for now. So that 
I can make sure the rest of the application works. 

'''
IMAGE_TMPL_DEF = '{:05d}.jpg'

from video_dataset import VideoDataset

class SocialGreetingDataSet(VideoDataset):

	class Data:
		def __init__(self, filename, action, history):
			self.filename = filename
			self.history = history
			self.action = action #label

	def __init__(self, root_path, transform, mode, segment_length, image_tmpl=IMAGE_TMPL_DEF):

		super().__init__(root_path, transform, mode, segment_length, image_tmpl=image_tmpl)

		self.action_dict = {
			'g':  [1],
			'a':  [1],
			'ga': [1],
			'za': [1],
			'zg': [1],
			'zga':[1],
			'z':    [0, 2],
			'none': [0, 2]}

		self.history = {
			0:[0],
			1:[0, 1],
			2:[1]
		}

		# generate all observation, hidden state, action combinations
		self.data = []

		print("obs:", self.obs_dict.keys())

		for obs_category in self.obs_dict.keys():
			for obs_sample in self.obs_dict[obs_category]:

				obs_file_dir = os.path.join(*[root_path, obs_category, obs_sample])

				for action in self.action_dict[obs_category]:
					for history in self.history[action]:
						self.data.append( self.Data(obs_file_dir, history, action) )
		
	def __getitem__(self, index):

		data = self.data[index]

		obs_x = self.parse_obs(data.filename)
		world_x = data.history
		action_y = data.action

		#print(type(obs_x), type(world_x), type(action_y))
		print(index, obs_x.size(), world_x, action_y)

		return obs_x, world_x, action_y


def create_dataloader(file_path, mode, batch_size=1, num_workers=16, max_length=8):

	# define transform function
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Compose([GroupMultiScaleCrop(224, [1, .875, .75, .66])]),
		#np.concatenate(img_group, axis=2)
		Stack(roll=(False)), # this is the culprit
		ToTorchFormatTensor(div=(True)),
		IdentityTransform(),
		])

	# setup path parameters
	assert mode in ["train", "validate", "evaluation"], "ERROR: mode must be either 'train', 'validate', or 'evaluation'"

	root_path = os.path.join(file_path, mode)

	shuffle = False
	if mode == "train":
		shuffle = True

	# create dataset
	dataset = SocialGreetingDataSet( root_path,
		image_tmpl=IMAGE_TMPL_DEF,
		transform=transform,
		mode=mode, 
		segment_length=max_length )

	# create dataloader
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers, 
		pin_memory = True)
