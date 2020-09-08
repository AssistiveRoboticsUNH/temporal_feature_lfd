import sys, os
import torch
import numpy as np
import pandas as pd

def eval(lfd_params, net):

	# Create DataLoaders
	#----------------

	eval_loader = lfd_params.create_dataloader(lfd_params, "evaluation", verbose=True)

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

	with torch.no_grad():

		for i, (obs, state, action, filename) in enumerate(eval_loader):

			# process visual observation data
			#obs = torch.reshape(obs, (-1, lfd_params.args.num_segments * 10, 3, 224,224))#obs.view(-1, max_length, 3, 224,224)
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

			for j, file in enumerate(filename):
				# add information to DataFrame
				rec_obs_label.append(file.split('/')[-2])
				rec_state.append(state[j].detach().cpu().numpy())
				rec_expected_action.append(action[j].detach().cpu().numpy())
				rec_observed_action.append(action_out[j])

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

		out_filename = os.path.join(lfd_params.args.output_dir, "output_"+lfd_params.args.save_id+".csv")
		df.to_csv(out_filename)

		print("Output placed in: "+out_filename)

if __name__ == '__main__':

	from parameter_parser import parse_model_args
	lfd_params = parse_model_args()

	from model.model import LfDNetwork
	net = LfDNetwork(lfd_params, is_training = False)

	eval(lfd_params, net)

