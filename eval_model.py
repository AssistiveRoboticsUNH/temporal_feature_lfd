import sys, os
import torch
import numpy as np
import pandas as pd

def eval(lfd_params, net):

	# Create DataLoaders
	#----------------

	eval_loader = lfd_params.create_dataloader(
		lfd_params.file_directory, 
		"evaluation", 
		batch_size=2,
		max_length=lfd_params.args.max_length,
		num_segments=lfd_params.args.num_segments,
		num_workers=1,
		verbose=True,
		)

	# Build Network
	#----------------

	# put model on GPU
	net = torch.nn.DataParallel(net, device_ids=lfd_params.args.gpus).cuda()
	net.eval()

	# define loss function
	#criterion = torch.nn.CrossEntropyLoss().cuda()
		
	# Evaluate Network
	#----------------
	rec_obs_label = []
	rec_state = []
	rec_expected_action = []
	rec_observed_action = []

	for i, (obs, state, action, filename) in enumerate(eval_loader):
		
		if(i % 100 == 0):
			print("iter: {:6d}/{:6d}".format(i, len(eval_loader)))

		# process visual observation data
		obs = torch.reshape(obs, (-1, lfd_params.args.num_segments * 10, 3, 224,224))#obs.view(-1, max_length, 3, 224,224)
		obs_x = torch.autograd.Variable(obs)

		# process hidden world data
		state_x = torch.autograd.Variable(state)

		# input shapes
		if (i == 0):
			print("obs_x: ", obs_x.shape)
			print("state_x: ", state_x.shape)
		
		# process action label
		action = action.cuda()
		action_y = torch.autograd.Variable(action)
		
		# compute output
		action_logits = net(obs_x, state_x)

		action_logits = action_logits.detach().cpu().numpy()
		action_out = np.argmax(action_logits, axis=1)

		for i, file in enumerate(filename):
			# add information to DataFrame
			rec_obs_label.append(file.split('/')[-2])
			rec_state.append(state[i])
			rec_expected_action.append(action[i])
			rec_observed_action.append(action_out[i])


		if(i % 100 == 0):
			print("iter: {:6d}/{:6d}".format(i, len(eval_loader)))

	# write output to file
	import datetime
	currentDT = datetime.datetime.now()
	df = pd.DataFrame({
			"obs_label":rec_obs_label,
			"state":rec_state,
			"expected_action":rec_expected_action,
			"observed_action":rec_observed_action,
		})
	out_filename = os.path.join(lfd_params.args.output_dir, "saved_model_"+lfd_params.args.app+"_"+currentDT.strftime("%Y-%m-%d_%H-%M-%S")+".csv")
	df.to_csv(out_filename)

if __name__ == '__main__':

	from parameter_parser import parse_model_args
	lfd_params = parse_model_args()

	from model.model import LfDNetwork
	net = LfDNetwork(lfd_params, is_training = False)

	eval(lfd_params, net)

