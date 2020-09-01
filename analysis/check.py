import torch
import torch.nn as nn
import numpy as np 
import random

from torch.utils.data import Dataset, DataLoader

class XORDataset(Dataset):
	def __init__(self):
		self.dataset = [ [1,1], [1,0], [0,1], [0,0] ]
		self.labelset = [  [0],   [1],   [1],   [0] ]

	def __getitem__(self, i):
		data  = torch.tensor([self.dataset[i]], dtype=torch.float)
		label = torch.tensor(self.labelset[i])
		return data, label

	def __len__(self):
		return len(self.labelset)


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = nn.Sequential(
			nn.Linear(2,2),
			nn.Tanh(),
			nn.Linear(2,2),
			nn.Tanh()
		)
		#self.lin = nn.Linear(2,2)

	def forward(self, inp):
		return self.lin(inp)

data_dict = {}


batch_size = 4
num_workers = 4

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

for run in range(5):

	net = Model()

	#net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
	net.train()

	# define loss function
	criterion = torch.nn.CrossEntropyLoss()#.cuda()

	# define optimizer
	params = list(net.parameters())
	optimizer = torch.optim.SGD(params, 0.05)
		
	# Train Network
	#----------------
	losses = []
	epoch = 5000
	with torch.autograd.detect_anomaly():
		for e in range(epoch):
			for n, (data, label) in enumerate(train_dl):

				optimizer.zero_grad()

				#data  = torch.tensor([dataset[i]], dtype=torch.float)
				#label = torch.tensor(labelset[i])

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
		for n, (data, label) in enumerate(eval_dl):
			#data  = torch.tensor([dataset[i]], dtype=torch.float)
			#label = labelset[i]

			data = torch.autograd.Variable(data)#.cuda()
			logits = net(data)

			out = np.argmax(logits.cpu().detach().numpy(), axis=1)

			#print(dataset[i], out, label)
	print("")

# show Losses
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd 
df = pd.DataFrame(data_dict)
df["avg"] = df.mean(axis=1)

plt.plot(df["avg"])
plt.savefig("analysis/fig/plt.png")
