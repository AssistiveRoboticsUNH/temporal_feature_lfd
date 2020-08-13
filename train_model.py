import sys
import torch

def train(lfd_params, net):

	# Create DataLoaders
	#----------------

	train_loader = lfd_params.create_dataloader(lfd_params.file_directory, "train")
	#validation_loader = lfd_params.create_dataloader(params, "validation")

	# Build Network
	#----------------

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

	for i, (obs, state, action) in enumerate(train_loader):
		if(i % 100 == 0):
			print("iter: {:6d}/{:6d}".format(i, len(train_loader)))

		# process visual observation data
		obs_x = torch.autograd.Variable(obs)

		# process hidden world data
		state_x = torch.autograd.Variable(state)

		# process action label
		action = action.cuda()
		action_y = torch.autograd.Variable(action)

		# compute output
		action_out = model.net(obs_x, state_x)

		loss = criterion(action_out, action_y)

		# compute gradient and do SGD step
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

	# save trained model
	import datetime
	currentDT = datetime.datetime.now()
	torch.save(model, "./saved_model_"+currentDT.strftime("%Y-%m-%d_%H-%M-%S")+".pt")

if __name__ == '__main__':

	from parameter_parser import parse_model_args
	lfd_params = parse_model_args()

	from model.model import LfDNetwork
	net = LfDNetwork(lfd_params, is_training = True)

	train(lfd_params, net)

