import os
import torch
import torch.nn as nn
from torch_geometric.nn.conv import RGCNConv
import torch_geometric.nn as gnn
import torch.nn.functional as F


class TemporalExtGCN(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 node_size=500, num_relations=1, output_size=4):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = os.path.join(filename, ".".join(["model", "temporal_gcn", "pt"]))

        # constants params
        self.num_relations = num_relations
        self.node_size = node_size
        self.hidden_size = 512
        self.output_size = output_size

        # define model vars
        self.gcn1 = RGCNConv(self.node_size, self.hidden_size, num_relations=self.num_relations)
        self.gcn2 = RGCNConv(self.hidden_size, self.hidden_size, num_relations=self.num_relations)
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
        edge_idx = edge_idx.long().cuda()
        edge_attr = edge_attr.cuda()
        batch = batch.cuda()

        x = F.relu(self.gcn1(x, edge_idx, edge_attr))
        x = F.relu(self.gcn2(x, edge_idx, edge_attr))
        x = gnn.global_add_pool(x, batch)

        x = self.fc(x)

        return x

    def save_model(self):
        torch.save(self.state_dict(), self.filename)
        print("TemporalExtLinear Linear model saved to: ", self.filename)

    def load_model(self, filename):
        assert os.path.exists(filename), "ERROR: temporal_ext_linear.py: Cannot locate saved model - "+filename

        print("Loading TemporalExtLinear from: " + filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint, strict=True)
        for param in self.parameters():
            param.requires_grad = False
