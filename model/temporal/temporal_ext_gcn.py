import os
import torch
import torch.nn as nn
from torch_geometric.nn.conv import RGCNConv, GCNConv
import numpy as np
from torch_geometric.data import Data


class TemporalExtGCN(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 node_size=11, num_relations=7, output_size=4):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename

        # constants params
        self.num_relations = num_relations
        self.node_size = node_size
        self.output_size = output_size

        # define model vars

        #CONSIDER STACKED (will need ReLU, check on actual ITR data)
        self.gcn = GCNConv(self.node_size, self.output_size)
        #self.gcn = RGCNConv(self.node_size, self.output_size, num_relations=self.num_relations)

        self.fc = nn.Linear(self.node_size * self.output_size, self.output_size)

        # load model parameters
        if not is_training:
            assert self.filename is not None, \
                "ERROR: temporal_ext_linear.py: filename must be defined when is_training is False"
            self.load_model(self.filename, self.gcn)
        else:
            print("TemporalExtLinear is training")

    # Defining the forward pass
    def forward(self, x):
        x = torch.reshape(x, (-1, self.node_size, self.node_size, self.num_relations))
        x = x.detach().cpu().numpy()[0]
        print("x:", x.shape)
        edge_idx = []
        #edge_idx = set()  # [2, num_edges] edge connections (COO format)
        edge_attr = [] # [1, num_edges] type of relationship (ITR)
        node_x = np.zeros((self.node_size, self.node_size))
        #node_x = np.arange(self.node_size).reshape(-1, 1)

        for i in range(self.node_size):
            node_x[i, i] = 1
            for j in range(self.node_size):
                for itr in range(self.num_relations):
                    if (x[i,j, itr] != 0):
                        edge_idx.append((i, j))
                        #edge_idx.add((i, j))
                        edge_attr.append(itr)

        #edge_idx = np.array(list(edge_idx)).T
        edge_idx = np.array(edge_idx).T
        edge_attr = np.array(edge_attr).reshape(1, -1)

        node_x = torch.autograd.Variable(torch.from_numpy(node_x).cuda()).float()
        edge_idx = torch.autograd.Variable(torch.from_numpy(edge_idx).cuda())
        edge_attr = torch.autograd.Variable(torch.from_numpy(edge_attr).cuda())

        #d = Data(x=node_x, edge_index=edge_idx)

        #assert False, "temporal_ext_gcn.py: Need to fromat the data for GCN"
        print("node_x:", node_x.shape, node_x.dtype)
        #print(node_x)
        print("edge_idx:", edge_idx.shape, edge_idx.dtype)
        #print(edge_idx)
        print("edge_attr:", edge_attr.shape, edge_attr.dtype)

        x = self.gcn(node_x, edge_idx)#, edge_attr)
        x = x.view((-1))
        x = torch.unsqueeze(x, dim=0)

        print("out:", x.shape)
        x = self.fc(x)

        print("out fc:", x.shape)


        return x

    def save_model(self, filename):
        torch.save(self.gcn.state_dict(), filename)
        print("TemporalExtLinear Linear model saved to: ", filename)

    def load_model(self, filename, var):
        assert os.path.exists(filename), "ERROR: temporal_ext_linear.py: Cannot locate saved model - "+filename

        print("Loading TemporalExtLinear from: " + filename)
        checkpoint = torch.load(filename)
        var.load_state_dict(checkpoint, strict=True)
        for param in var.parameters():
            param.requires_grad = False
