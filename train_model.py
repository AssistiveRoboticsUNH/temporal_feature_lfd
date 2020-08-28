import sys, os
import torch
from copy import deepcopy

def train(lfd_params, model):

	# Create DataLoaders
	#----------------

	train_loader = lfd_params.create_dataloader(lfd_params, "train")
		
	#validation_loader = lfd_params.create_dataloader(params, "validation")

	# Build Network
	#----------------

	# put model on GPU
	net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
	net.train()
	a2 = deepcopy(net.state_dict())

	# define loss function
	criterion = torch.nn.CrossEntropyLoss().cuda()

	# define optimizer
	params = list(net.parameters())
	optimizer = torch.optim.SGD(params,
								lfd_params.args.lr,
								momentum=lfd_params.args.momentum,
								weight_decay=lfd_params.args.weight_decay)
		
	# Train Network
	#----------------
	with autograd.detect_anomaly():

		epoch = lfd_params.args.epochs
		for e in range(epoch):
			for i, data_packet in enumerate(train_loader):

				obs, state, action = data_packet

				# process visual observation data
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

	# save trained model parameters
	out_filename = lfd_params.generate_modelname()
	#net.save_model_params(out_filename)
	print("\nCHECK:")

	b2 = net.state_dict()
	c2 = model.state_dict()
	print("a2 == c2", str(a2) == str(c2))
	print("b2 == c2", str(b2) == str(c2))

	for i, data in enumerate([a2,b2,c2]):
		f = open(str(i)+"_out", 'w')
		f.write(str(data))
		f.close()

	model.save_model()
	#torch.save(net.state_dict(), out_filename)

	return out_filename

if __name__ == '__main__':

	from parameter_parser import parse_model_args
	lfd_params = parse_model_args()

	from model.model import LfDNetwork
	net = LfDNetwork(lfd_params, is_training = True)

	train(lfd_params, net)

