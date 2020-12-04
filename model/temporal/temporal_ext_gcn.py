import os
import torch
import torch.nn as nn
from torch_geometric.nn.conv import RGCNConv, GCNConv
from torch_geometric.nn import global_add_pool
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
        self.hidden_size = 32
        self.output_size = output_size

        # define model vars

        # CONSIDER STACKED (will need ReLU, check on actual ITR data)
        # self.gcn = GCNConv(self.node_size, self.output_size)
        self.gcn = RGCNConv(self.node_size, self.hidden_size, num_relations=self.num_relations)

        # print("temp_ext_gcn.py:", self.node_size, int(self.node_size/2) * self.output_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # load model parameters
        if not is_training:
            assert self.filename is not None, \
                "ERROR: temporal_ext_linear.py: filename must be defined when is_training is False"
            self.load_model(self.filename, self.gcn)
        else:
            print("TemporalExtLinear is training")

    def forward(self, x):
        node_x, edge_idx, edge_attr, batch = x.x, x.edge_index, x.edge_attr, x.batch

        node_x = node_x.float().cuda()
        edge_idx = edge_idx.cuda()
        edge_attr = edge_attr.cuda()
        batch = batch.cuda()

        print("temp_ext_gcn node_x:", node_x.shape, type(node_x), node_x.dtype)
        print("temp_ext_gcn edge_idx:", edge_idx.shape, type(edge_idx), edge_idx.dtype)
        print("temp_ext_gcn edge_attr:", edge_attr.shape, type(edge_attr), edge_attr.dtype)

        x = self.gcn(node_x, edge_idx, edge_attr)
        x = F.relu(x)
        print("out:", x.shape, x.dtype)
        x = global_add_pool(x, batch)
        print("out1:", x.shape, x.dtype)
        print(x)
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
