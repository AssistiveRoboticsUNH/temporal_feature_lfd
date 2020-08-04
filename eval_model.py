import net

def eval(model):

	# Create Dataset
	#----------------




	# Create DataLoaders
	#----------------

	eval_loader

	# Build Network
	#----------------

	# define loss function
	criterion = torch.nn.CrossEntropyLoss().cuda()

	# Evaluate Network
	#----------------

	with torch.no_grad():
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



			# measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            



	# log evaulation model
	import datetime
	currentDT = datetime.datetime.now()
	#torch.save(model, "./saved_model_"+currentDT.strftime("%Y-%m-%d_%H-%M-%S")+".pt")



