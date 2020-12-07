import os
import torch
import torch.nn as nn
from torch_geometric.nn.conv import RGCNConv, GCNConv, GINConv
import torch_geometric.nn as gnn
import numpy as np
from torch_geometric.data import Data
import torch.nn.functional as F


class TemporalExtGCN(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 node_size=500, num_relations=7, output_size=4):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename

        # constants params
        self.num_relations = num_relations
        self.node_size = node_size
        self.hidden_size = 64
        self.output_size = output_size

        # define model vars

        # CONSIDER STACKED (will need ReLU, check on actual ITR data)
        #self.gcn = GCNConv(self.node_size, self.hidden_size)

        #self.gcn1 = GCNConv(self.node_size, self.hidden_size)
        #self.gcn2 = GCNConv(self.hidden_size, self.hidden_size)
        #self.gcn3 = GCNConv(self.hidden_size, self.hidden_size)
        #self.gcn4 = GCNConv(self.hidden_size, self.hidden_size)

        self.gcn1 = RGCNConv(self.node_size, self.hidden_size, num_relations=self.num_relations)
        self.gcn2 = RGCNConv(self.hidden_size, self.hidden_size, num_relations=self.num_relations)
        self.gcn3 = RGCNConv(self.hidden_size, self.hidden_size, num_relations=self.num_relations)
        self.gcn4 = RGCNConv(self.hidden_size, self.hidden_size, num_relations=self.num_relations)
        #self.densegcn = gnn.DenseGCNConv(self.hidden_size, self.output_size)
        #nn1 = nn.Sequential(nn.Linear(self.node_size, self.hidden_size), nn.ReLU(), nn.Linear(self.hidden_size, self.hidden_size))
        #self.gcn = GINConv(nn1)

        # print("temp_ext_gcn.py:", self.node_size, int(self.node_size/2) * self.output_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # load model parameters
        if not is_training:
            assert self.filename is not None, \
                "ERROR: temporal_ext_linear.py: filename must be defined when is_training is False"
            self.load_model(self.filename)
        else:
            print("TemporalExtLinear is training")

    def forward(self, x):
        x, edge_idx, edge_attr, batch = x.x, x.edge_index, x.edge_attr, x.batch

        x = x.float().cuda()
        edge_idx = edge_idx.cuda()
        edge_attr = edge_attr.cuda()
        batch = batch.cuda()

        print("temp_ext_gcn node_x:", x.shape, type(x), x.dtype)
        print("temp_ext_gcn edge_idx:", edge_idx.shape, type(edge_idx), edge_idx.dtype)
        print("temp_ext_gcn edge_attr:", edge_attr.shape, type(edge_attr), edge_attr.dtype)

        x = F.relu(self.gcn1(x, edge_idx, edge_attr))
        x = F.relu(self.gcn2(x, edge_idx, edge_attr))
        x = F.relu(self.gcn3(x, edge_idx, edge_attr))
        x = F.relu(self.gcn4(x, edge_idx, edge_attr))


        print("out:", x.shape, x.dtype)
        print("batch:", batch)
        x = gnn.global_add_pool(x, batch)
        #x = gnn.global_mean_pool(x, batch)
        #x = gnn.global_max_pool(x, batch)
        #print(x)

        x = self.fc(x)
        print("out fc:", x.shape)

        return x

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("TemporalExtLinear Linear model saved to: ", filename)

    def load_model(self, filename):
        assert os.path.exists(filename), "ERROR: temporal_ext_linear.py: Cannot locate saved model - "+filename

        print("Loading TemporalExtLinear from: " + filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint, strict=True)
        for param in self.parameters():
            param.requires_grad = False
