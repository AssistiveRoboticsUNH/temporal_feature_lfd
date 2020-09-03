import sys, os
import torch

def train(lfd_params, model):

	# Create DataLoaders
	#----------------

	train_loader = lfd_params.create_dataloader(lfd_params, "train")
		
	#validation_loader = lfd_params.create_dataloader(params, "validation")

	# Build Network
	#----------------

	# put model on GPU
	params = list(model.parameters())
	net = torch.nn.DataParallel(model, device_ids=lfd_params.args.gpus).cuda()
	net.train()

	# define loss function
	criterion = torch.nn.CrossEntropyLoss().cuda()

	# define optimizer
	optimizer = torch.optim.SGD(params,
								lfd_params.args.lr,
								momentum=lfd_params.args.momentum,
								weight_decay=lfd_params.args.weight_decay)
		
	loss_record = []

	# Train Network
	#----------------
	with torch.autograd.detect_anomaly():

		epoch = lfd_params.args.epochs
		for e in range(epoch):
			for i, data_packet in enumerate(train_loader):

				obs, state, action = data_packet

				#obs = obs.view(lfd_params.args.batch_size, -1, 3, 224, 224)[:, 4]
				#obs = obs.view(lfd_params.args.batch_size, 3, 224, 224)

				#print("obs:", obs.shape)
				#print("obs0:", obs[0])
				#print("obs1:", obs[1])
				#print("obs2:", obs[2])




				# process visual observation data
				obs_x = obs#torch.autograd.Variable(obs)

				# process hidden world data
				state_x = state#torch.autograd.Variable(state)

				# input shapes
				if (e == 0 and i == 0):
					print("obs_x: ", obs_x.shape)
					print("state_x: ", state_x.shape)
				
				# process action label
				action = action.cuda()
				action_y = action#torch.autograd.Variable(action)
				
				# compute output
				action_logits = net(obs_x, state_x)

				# get loss
				loss = criterion(action_logits, action_y)
				loss.backward()

				# optimize SGD
				optimizer.step()
				optimizer.zero_grad()

				if(i % 100 == 0):
					print("epoch: {:3d}/{:3d},  iter: {:6d}/{:6d}".format(e, epoch, i, len(train_loader)))
					print("loss:", loss.cpu().detach().numpy())
					print("expected:", action.cpu().detach().numpy())
					print("output:", action_logits.cpu().detach().numpy())

				loss_record.append(loss.cpu().detach().numpy())

	# save trained model parameters
	out_filename = lfd_params.generate_modelname()

	model.save_model()
	#torch.save(net.state_dict(), out_filename)

	import matplotlib
	import matplotlib.pyplot as plt

	plt.plot(loss_record)
	plt.savefig("analysis/fig/loss_record.png")


	return out_filename

if __name__ == '__main__':

	from parameter_parser import parse_model_args
	lfd_params = parse_model_args()

	from model.model import LfDNetwork
	net = LfDNetwork(lfd_params, is_training = True)

	train(lfd_params, net)

