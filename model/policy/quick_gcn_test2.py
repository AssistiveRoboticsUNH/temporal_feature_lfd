from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader



import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print("x.shape:", x.shape, x.dtype)
        #print(torch.max(x, dim=1))
        print("edge_index.shape:", edge_index.shape,  edge_index.dtype)
        x = self.conv1(x, edge_index)
        print("conv1.shape:", x.shape, x.dtype)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        print("conv2.shape:", x.shape, x.dtype)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("dataset:", dataset)
print("dataset[0]:", dataset[0])


model.train()

for epoch in range(200):
    for batch in loader:
        print(batch)
        optimizer.zero_grad()
        out = model(batch.to(device))
        print("out:", out.shape, batch.y.shape)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))