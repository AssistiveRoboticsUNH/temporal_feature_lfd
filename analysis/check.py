import torch
import torch.nn as nn
import numpy as np 
import random

dataset = [ [1,1], [1,0], [0,1], [0,0] ]
labelset = [ 0, 1, 1, 0 ]


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = nn.Linear(2,1)

	def forward(self, inp):
		return self.lin(inp)

net = Model()

net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
net.train()

# define loss function
criterion = torch.nn.CrossEntropyLoss().cuda()

# define optimizer
params = list(net.parameters())
optimizer = torch.optim.SGD(params, 0.01)
	
# Train Network
#----------------
epoch = 10
with torch.autograd.detect_anomaly():
	for e in range(epoch):
		i = random.randint(0, 3)
		data, label = torch.tensor(dataset[i]), torch.tensor(labelset[i])

		data = torch.autograd.Variable(data).cuda()
		label = torch.autograd.Variable(label).cuda()
		
		# compute output
		logits = net(data)

		# get loss
		loss = criterion(logits, label)
		loss.backward()

		# optimize SGD
		optimizer.step()
		optimizer.zero_grad()

		print("loss:", loss.cpu().detach().numpy())
		print("expected:", label.cpu().detach().numpy())
		print("output:", logits.cpu().detach().numpy())

