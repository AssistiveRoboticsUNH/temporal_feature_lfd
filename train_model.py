import sys, os
import torch

def train(lfd_params, net):

	# Create DataLoaders
	#----------------

	train_loader = lfd_params.create_dataloader(
		lfd_params.file_directory, 
		"train", 
		num_segments=lfd_params.args.num_segments,
		)
	#validation_loader = lfd_params.create_dataloader(params, "validation")

	# Build Network
	#----------------

	# put model on GPU
	net = torch.nn.DataParallel(net, device_ids=lfd_params.args.gpus).cuda()
	net.train()

	# define loss function
	criterion = torch.nn.CrossEntropyLoss().cuda()

	# define optimizer
	lr = 0.01
	momentum = 0.9
	weight_decay = 0.0005

	params = list(net.parameters())
	optimizer = torch.optim.SGD(params,
								lr,
								momentum=momentum,
								weight_decay=weight_decay)
		
	# Train Network
	#----------------

	epoch = 10
	for e in range(epoch):
		for i, (obs, state, action) in enumerate(train_loader):

			# process visual observation data
			obs = torch.reshape(obs, (-1, lfd_params.args.num_segments, 3, 224,224))
			obs_x = torch.autograd.Variable(obs)

			# process hidden world data
			state_x = torch.autograd.Variable(state)

			# input shapes
			if (e == 0 and i == 0):
				print("obs_x: ", obs_x.shape)
				print("state_x: ", state_x.shape)
			
			# process action label
			action = action.cuda()
			action_y = torch.autograd.Variable(action)
			
			# compute output
			action_logits = net(obs_x, state_x)

			loss = criterion(action_logits, action_y)

			# compute gradient and do SGD step
			loss.backward()

			optimizer.step()
			optimizer.zero_grad()

			if(i % 100 == 0):
				print("epoch: {:3d}/{:3d},  iter: {:6d}/{:6d}".format(e, epoch, i, len(train_loader)))

	# save trained model
	import datetime
	currentDT = datetime.datetime.now()
	out_filename = os.path.join(lfd_params.args.model_dir, "saved_model_"+lfd_params.args.app+"_"+currentDT.strftime("%Y-%m-%d_%H-%M-%S")+".pt")
	torch.save(net, out_filename)

if __name__ == '__main__':

	from parameter_parser import parse_model_args
	lfd_params = parse_model_args()

	from model.model import LfDNetwork
	net = LfDNetwork(lfd_params, is_training = True)

	train(lfd_params, net)

