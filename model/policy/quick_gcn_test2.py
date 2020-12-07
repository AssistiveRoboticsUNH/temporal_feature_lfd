import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, Dataset

class MyDataset(Dataset):
    def __init__(self):

        data1 = np.array([[1,0], [0,1]])
        edge1 = np.array([[0,1], [1,0]]).T
        att1 = np.array([0, 0])

        data2 = np.array([[1,0], [0,1], [1,0]])
        edge2 = np.array([[0, 1], [1, 0]]).T
        att2 = np.array([0, 0])

        self.data = [
                Data(x=data1, edge_index=edge1, edge_attr=att1, y=0),
                Data(x=data2, edge_index=edge2, edge_attr=att2, y=1)
        ]

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

#path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'MUTAG')
#dataset = TUDataset(path, name='MUTAG').shuffle()
#test_dataset = dataset[:len(dataset) // 10]
#train_dataset = dataset[len(dataset) // 10:]

dataset = MyDataset()
train_dataset = dataset
test_dataset = dataset

test_loader = DataLoader(test_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        num_features = dataset.num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        #print('x1:', x.shape)
        x = F.relu(self.conv1(x, edge_index))
        #print('x2:', x.shape)
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        #print('x2.1:', x.shape)
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        #print('x2.2:', x.shape)
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        print("x3:", x.shape)
        x = global_add_pool(x, batch)
        #x = global_mean_pool(x, batch)
        #x = global_max_pool(x, batch)
        print('x4:', x.shape)
        x = F.relu(self.fc1(x))
        #print('x5:', x.shape)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        #print('x6:', x.shape)
        x = F.log_softmax(x, dim=-1)
        #print('x7:', x.shape)
        #print('\n')
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        print('out:', output.shape, data.y.shape)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        #print(data.batch)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


for epoch in range(1, 101):
    train_loss = train(epoch)
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                       train_acc, test_acc))
