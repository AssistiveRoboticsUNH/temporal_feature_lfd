import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os, sys

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
IMAGE_TMPL_DEF = 'image_{:05d}.jpg'

from video_dataset import VideoDataset

class SocialGreetingDataSet(VideoDataset):

	class Data:
		def __init__(self, filename, history, action):
			self.filename = filename
			self.history = history
			self.action = action #label

	def __init__(self, 
			root_path, 
			mode, 
			image_tmpl=IMAGE_TMPL_DEF, 
			num_segments=3,
			verbose=False,
		):

		super().__init__(root_path, mode, image_tmpl=image_tmpl, flip=False, num_segments=3)

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
		self.verbose = verbose

		#print("obs:", self.obs_dict.keys())

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
		#print(index, obs_x.size(), world_x, action_y)
		if (not self.verbose):
			return obs_x, world_x, action_y
		else:
			return obs_x, world_x, action_y, data.filename


def create_dataloader(lfd_params, mode):
	file_path = lfd_params.file_directory
	full_sample = lfd_params.args.use_ditrl
	batch_size = lfd_params.args.batch_size
	num_workers = lfd_params.args.num_dl_workers
	num_segments = lfd_params.args.num_segments

	# setup path parameters
	assert mode in ["train", "validate", "evaluation"], "ERROR: mode must be either 'train', 'validate', or 'evaluation'"

	root_path = os.path.join(file_path, mode)

	shuffle = False
	if mode == "train":
		shuffle = True

	# create dataset
	dataset = SocialGreetingDataSet( 
		root_path,
		image_tmpl=IMAGE_TMPL_DEF,
		mode=mode, 
		num_segments=num_segments,
		verbose=verbose, 
		)

	# create dataloader
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers, 
		pin_memory = True)
