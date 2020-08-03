import argparse
import os

class Parameters:
	def __init__(self, args):

		self.args = args

		if(self.args.app == "bi"):
			self.setup_social_greeting()
		elif(self.args.app == "bs"):
			self.setup_block_stacking()

		self.checkpoint_file = ""

	def setup_social_greeting(self):
		self.file_directory = "~/datasets/SocialGreeting/"
		self.num_actions = 3

		from ops.social_greeting_dl import create_dataloader
		self.create_dataloader = create_dataloader



	def setup_block_stacking(self):
		self.file_directory = ""
		self.num_actions = 7

		#rom social_greeting_dl import create_dataloader
		#self.create_dataloader = create_dataloader

def parse_model_args():
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	
	# model command line args
	parser.add_argument('app', help='the checkpoint file to use with the model', choices=['bi', 'bs'])

	# whether the model should use D-ITR-L or not
	parser.add_argument('--use_ditrl', dest='use_ditrl', action='store_true')
	parser.set_defaults(use_ditrl=False)
	parser.add_argument('--bottleneck_size', type=int, help='if using D-ITR-L what bottleneck size.')

	# whether the model is being trained
	parser.add_argument('train', default=False, help='gpu to run on')
	parser.add_argument('--modelname', default=False, help='name of saved features to store or load')
	parser.add_argument('--gpu', default="0", help='which gpu to run on')

	# if trained then require:
	parser.add_argument('--epochs', type=int, default=False, help='gpu to run on')
	parser.add_argument('--lr', type=float, default=False, help='gpu to run on')

	return Parameters(parser.parse_args())
