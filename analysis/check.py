import torch
import torch.nn as nn
import numpy as np 
import random

dataset = [ [1,1], [1,0], [0,1], [0,0] ]
labelset = [  [0],   [1],   [1],   [0] ]


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = nn.Sequential(
			nn.Linear(2,2),
			nn.Linear(2,2)
		)
		#self.lin = nn.Linear(2,2)

	def forward(self, inp):
		return self.lin(inp)

data_dict = {}

for run in range(5):

	net = Model()

	#net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
	net.train()

	# define loss function
	criterion = torch.nn.CrossEntropyLoss()#.cuda()

	# define optimizer
	params = list(net.parameters())
	optimizer = torch.optim.SGD(params, 0.1)
		
	# Train Network
	#----------------
	losses = []
	epoch = 200
	with torch.autograd.detect_anomaly():
		for e in range(epoch):
			i = random.randint(0, 3)

			optimizer.zero_grad()

			data  = torch.tensor([dataset[i]], dtype=torch.float)
			label = torch.tensor(labelset[i])

			data = torch.autograd.Variable(data)#.cuda()
			label = torch.autograd.Variable(label)#.cuda()
			
			# compute output
			logits = net(data)

			# get loss
			loss = criterion(logits, label)
			loss.backward()

			# optimize SGD

			optimizer.step()
			

			losses.append(loss.cpu().detach().numpy())
	data_dict["run_"+str(run)] = losses

	# eval model
	net.eval()
	with torch.no_grad():
		for i in range(len(dataset)):
			data  = torch.tensor([dataset[i]], dtype=torch.float)
			label = labelset[i]

			data = torch.autograd.Variable(data)#.cuda()
			logits = net(data)

			out = np.argmax(logits.cpu().detach().numpy())

			print(dataset[i], out, label)
	print("")

# show Losses
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd 
df = pd.DataFrame(data_dict)
df["avg"] = df.mean(axis=1)

plt.plot(df["avg"])
plt.savefig("analysis/fig/plt.png")
