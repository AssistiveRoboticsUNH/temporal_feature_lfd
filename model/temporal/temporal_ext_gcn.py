import os
import torch
import torch.nn as nn
from torch_geometric.nn.conv import RGCNConv


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
        self.gcn = RGCNConv(self.node_size, self.output_size, num_relations=self.num_relations)

        #self.fc = nn.Linear(self.input_size, self.output_size)

        # load model parameters
        if not is_training:
            assert self.filename is not None, \
                "ERROR: temporal_ext_linear.py: filename must be defined when is_training is False"
            self.load_model(self.filename, self.gcn)
        else:
            print("TemporalExtLinear is training")

    # Defining the forward pass
    def forward(self, x):
        edges = torch.reshape(x, (-1, self.node_size, self.node_size, self.num_relations))

        #assert False, "temporal_ext_gcn.py: Need to fromat the data for GCN"

        x = self.gcn(x, edges)
        #x = self.fc(x)

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
