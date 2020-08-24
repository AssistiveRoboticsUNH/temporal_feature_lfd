import sys
import torch

def eval(lfd_params, net):

	# Create DataLoaders
	#----------------

	eval_loader = lfd_params.create_dataloader(
		lfd_params.file_directory, 
		"evaluation", 
		max_length=lfd_params.args.max_length,
		num_workers=2)

	# Build Network
	#----------------

	# put model on GPU
	net = torch.nn.DataParallel(net, device_ids=lfd_params.args.gpus).cuda()
	net.eval()
		
	# Evaluate Network
	#----------------

	for i, (obs, state, action) in enumerate(eval_loader):
		if(i % 100 == 0):
			print("iter: {:6d}/{:6d}".format(i, len(eval_loader)))

		# process visual observation data
		max_length = 8
		obs = torch.reshape(obs, (-1, max_length, 3, 224,224))#obs.view(-1, max_length, 3, 224,224)
		obs_x = torch.autograd.Variable(obs)

		# process hidden world data
		state_x = torch.autograd.Variable(state)

		print("obs_x: ", obs_x.size())
		print("state_x: ", state_x.size())
		
		# process action label
		action = action.cuda()
		action_y = torch.autograd.Variable(action)
		
		# compute output
		action_out = net(obs_x, state_x)

if __name__ == '__main__':

	from parameter_parser import parse_model_args
	lfd_params = parse_model_args()

	from model.model import LfDNetwork
	net = LfDNetwork(lfd_params, is_training = False)

	eval(lfd_params, net)

