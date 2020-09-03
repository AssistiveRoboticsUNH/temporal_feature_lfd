import torch
import torch.nn as nn
import numpy as np 
import random

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class XORDataset(Dataset):
	def __init__(self):
		self.dataset = [ [1,1], [1,0], [0,1], [0,0] ]
		self.labelset = [  0,   1,   1,   0 ]

	def __getitem__(self, i):
		data  = torch.tensor(self.dataset[i], dtype=torch.float)
		label = torch.tensor(self.labelset[i], dtype=torch.float)
		return data, label

	def __len__(self):
		return len(self.labelset)


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin1 = nn.Linear(2,2)
		self.lin2 = nn.Linear(2,1)
		'''
		self.lin = nn.Sequential(
			nn.Linear(2,2),
			#nn.Sigmoid(),
			nn.Linear(2,1),
			#nn.Sigmoid()
		)
		#self.lin = nn.Linear(2,2)
		'''
	def forward(self, x):
		x = self.lin1(x)
		x = F.sigmoid(x)
		x = self.lin2(x)
		return x
		#return self.lin(x)

data_dict = {}


batch_size = 1
num_workers = 16

dataset = XORDataset()
train_dl = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers, 
		pin_memory = True)
eval_dl = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers, 
		pin_memory = True)

for run in range(1):

	net = Model()

	#net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
	net.train()

	# define loss function
	criterion = torch.nn.MSELoss()#torch.nn.CrossEntropyLoss()#.cuda()

	# define optimizer
	#params = list(net.parameters())
	optimizer = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9)
		
	# Train Network
	#----------------
	losses = []
	epoch = 2000
	with torch.autograd.detect_anomaly():
		for e in range(epoch):
			#print("e: {:4d}/{:4d}".format(e, epoch))
			for n, (data, label) in enumerate(train_dl):

				optimizer.zero_grad()
				
				# compute output
				logits = net(data)

				# get loss
				loss = criterion(logits, label)
				loss.backward()

				# optimize SGD
				optimizer.step()
				
				loss_v = loss.cpu().detach().numpy()
				losses.append(loss_v)
				print("loss_v:", loss_v)

		#if i % 500 == 0:
		#	print("Epoch: {0}, Loss: {1}, ".format(i, loss.data.numpy()) )

	data_dict["run_"+str(run)] = losses

	# eval model
	#net.eval()
	#with torch.no_grad():
	for n, (data, label) in enumerate(eval_dl):
		#data  = torch.tensor([dataset[i]], dtype=torch.float)
		#label = labelset[i]

		data = data
		logits = net(data)

		out = np.argmax(logits.cpu().detach().numpy(), axis=1)

		print(data, out, label)
	print("")

print("params:")
print(list(net.parameters()))

# show Losses
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd 
df = pd.DataFrame(data_dict)
df["avg"] = df.mean(axis=1)

plt.plot(df["avg"])
plt.savefig("analysis/fig/plt.png")
