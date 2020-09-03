import argparse
import os
import datetime

ROOT_DIR = "/home/mbc2004/"

class Parameters:
	def __init__(self, args):

		self.args = args

		if(self.args.app == "bi"):
			self.setup_social_greeting()
		elif(self.args.app == "bs"):
			self.setup_block_stacking()

		if(not os.path.exists(self.args.model_dir)):
			os.makedirs(self.args.model_dir)

		if(not os.path.exists(self.args.output_dir)):
			os.makedirs(self.args.output_dir)

		if(self.args.save_id != ""):
			filename = os.path.join(self.args.model_dir, "saved_model_"+self.args.save_id+"."+"backbone"+".pt")
			if (os.path.exists(filename)):
				print("file found: ", filename)
				self.args.backbone_modelname = filename

			filename = os.path.join(self.args.model_dir, "saved_model_"+self.args.save_id+"."+"ext"+".pt")
			if (os.path.exists(filename)):
				print("file found: ", filename)
				self.args.ext_modelname = filename

			filename = os.path.join(self.args.model_dir, "saved_model_"+self.args.save_id+"."+"policy"+".pt")
			if (os.path.exists(filename)):
				print("file found: ", filename)
				self.args.policy_modelname = filename

		else:
			self.generate_save_id()

		print(self.args)


	def setup_social_greeting(self):
		self.file_directory = os.path.join(ROOT_DIR, "datasets/SocialGreeting/frames/")
		self.num_actions = 3
		self.num_hidden_state_params = 1

		self.use_aud = False
		self.checkpoint_file = os.path.join(ROOT_DIR, "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
		self.trained_checkpoint_file = os.path.join(ROOT_DIR, "models/social_greeting_tsm.pth")

		from social_greeting_dl import create_dataloader
		self.create_dataloader = create_dataloader

	def setup_block_stacking(self):
		self.file_directory = os.path.join(ROOT_DIR, "datasets/BlockConstruction/frames/")
		self.num_actions = 7
		self.num_hidden_state_params = 1

		self.use_aud = False
		self.checkpoint_file = os.path.join(ROOT_DIR, "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
		self.trained_checkpoint_file = os.path.join(ROOT_DIR, "models/block_construction_tsm.pth")

		from block_construction_dl import create_dataloader
		self.create_dataloader = create_dataloader

	def generate_save_id(self):
		if(self.args.save_id==""):
			currentDT = datetime.datetime.now()
			use_ditrl = "ditrl_" if self.args.use_ditrl else ""
			use_trim = "trim_" if self.args.trim_model else ""
			self.args.save_id = self.args.app+"_"+use_ditrl+use_trim+currentDT.strftime("%Y-%m-%d_%H-%M-%S")

	def generate_modelname(self, section="null"):
		self.generate_save_id()
		return os.path.join(self.args.model_dir, "saved_model_"+self.args.save_id+"."+section+".pt")

	def generate_backbone_modelname(self):
		return self.generate_modelname(section="backbone")

	def generate_ext_modelname(self):
		return self.generate_modelname(section="ext")

	def generate_policy_modelname(self):
		return self.generate_modelname(section="policy")
		

def parse_model_args():
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	
	# model command line args
	parser.add_argument('app', help='the checkpoint file to use with the model', choices=['bi', 'bs'])

	# whether the model should use D-ITR-L or not
	parser.add_argument('--ditrl', dest='use_ditrl', action='store_true', help='flag denotes that D-ITR-L should be applied')
	parser.set_defaults(use_ditrl=False)
	parser.add_argument('--trim', dest='trim_model', action='store_true', help='flag denotes that Model should be trained on observations only, and should not be used to generate a policy')
	parser.set_defaults(trim_model=False)
	parser.add_argument('--bottleneck_size', type=int, default=128, help='if using D-ITR-L what bottleneck size.')

	# whether the model is being trained
	parser.add_argument('--model_dir', default="saved_models")
	parser.add_argument('--output_dir', default="csv_output")

	parser.add_argument('--save_id', default="", help='model_id to restore')
	parser.add_argument('--pretrain_modelname', default="/home/mbc2004/models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth", help='load the backbone model features from this file; these features can be fine-tuned and are not fixed')
	parser.add_argument('--backbone_modelname', default=False, help='load the backbone model features from this file; these features are fixed when this parameter is present')
	parser.add_argument('--ext_modelname', default=False, help='load the D-ITR-L model features from this file; these features are fixed when this parameter is present')
	parser.add_argument('--policy_modelname', default=False, help='load the Policy model features from this file; these features are fixed when this parameter is present')
	parser.add_argument('--gpus', nargs='+', type=int, default=None)

	# if trained then require:
	parser.add_argument('--num_dl_workers', type=int, default=8, help='the number of workers for the DataLoader')
	parser.add_argument('--batch_size', type=int, default=1, help='the number of segments to split a clip into')
	parser.add_argument('--num_segments', type=int, default=8, help='the number of segments to split a clip into')
	parser.add_argument('--fix_stride', type=int, default=5, help='the number of segments to split a clip into')
	parser.add_argument('--max_length', type=int, default=8, help='the length of a clip')

	parser.add_argument('--epochs', type=int, default=25, help='gpu to run on')
	parser.add_argument('--lr', type=float, default=0.02, help='gpu to run on')
	parser.add_argument('--weight_decay', type=float, default=0.0005, help='the length of a clip')
	parser.add_argument('--momentum', type=float, default=0.9, help='the length of a clip')


	return Parameters(parser.parse_args())
