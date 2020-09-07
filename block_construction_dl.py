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

class BlockConstructionDataSet(VideoDataset):

	class Data:
		def __init__(self, filename, history, action):
			self.filename = filename
			self.history = history
			self.action = action #label

	def __init__(self, 
			root_path, 
			mode, 
			full_sample,
			image_tmpl=IMAGE_TMPL_DEF, 
			num_segments=3,
			verbose=False,
			fix_stride=1,
			trim=False,
		):

		super().__init__(root_path, mode, full_sample, image_tmpl=image_tmpl, fix_stride=fix_stride, num_segments=num_segments)


		self.action_dict = {
			'r':  [0],
			'g':  [1],
			'b':  [2],
			#'gb': [3],
			#'bg': [4],
			#'rr': [5],
			#'rrr':[6]
			}

		self.history = {
			0:[0,2],
			1:[0],
			2:[0],
			3:[1],
			4:[1],
			5:[2],
			6:[2]
		}

		# generate all observation, hidden state, action combinations
		self.data = []
		self.verbose = verbose

		#print("obs:", self.obs_dict.keys())

		if (trim):
			for obs_category in self.obs_dict.keys():
				if obs_category in self.action_dict:

					for obs_sample in self.obs_dict[obs_category]:
						obs_file_dir = os.path.join(*[root_path, obs_category, obs_sample])
						for action in self.action_dict[obs_category]:
							self.data.append( self.Data(obs_file_dir, 0, action) )

		else:
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
		print("IS VERBOSE:", self.verbose)

		if (not self.verbose):
			return obs_x, world_x, action_y
		else:
			return obs_x, world_x, action_y, data.filename


def create_dataloader(lfd_params, mode, shuffle=None, verbose=None):
	# setup path parameters
	assert mode in ["train", "validate", "evaluation"], "ERROR: mode must be either 'train', 'validate', or 'evaluation'"
	is_training = (mode == "train")

	#root_path = os.path.join(lfd_params.file_directory, mode)
	root_path = os.path.join(lfd_params.file_directory, "train")

	# create dataset
	dataset = BlockConstructionDataSet( root_path,
		image_tmpl=IMAGE_TMPL_DEF,
		mode=mode, 
		num_segments=lfd_params.args.num_segments,
		verbose=not is_training if verbose==None else verbose, 
		full_sample=lfd_params.args.use_ditrl,
		fix_stride =lfd_params.args.fix_stride,
		trim=lfd_params.args.trim_model,
	)

	# create dataloader
	return DataLoader(
		dataset,
		batch_size=lfd_params.args.batch_size,
		shuffle=is_training if shuffle==None else shuffle,
		num_workers=lfd_params.args.num_dl_workers, 
		pin_memory = True)
