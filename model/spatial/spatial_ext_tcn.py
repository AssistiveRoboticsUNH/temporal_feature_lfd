import os
import torch
import torch.nn as nn

from torch.autograd import Variable
from .tcn import TemporalConvNet

# takes the output of a bottleneck filter and uses a max consensus and linear layer on the resultant features.


class SpatialExtTCN(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 input_size=128, output_size=8, consensus=None, dense_data=False, reshape_output=False):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        #self.filename = filename
        self.consensus = consensus
        self.reshape_output = reshape_output

        self.filename = os.path.join(filename, ".".join(["model", "spatial_tcn", "pt"]))

        assert self.consensus in [None, "max", "avg", "flat"], \
            "ERROR: spatial_ext_linear.py: consensus must be either None, 'max', 'avg', of 'flat'"
        self.dense_data = dense_data

        # constants params
        self.input_size = input_size
        self.hidden_size = 32
        self.num_layers = 1
        self.output_size = output_size

        # define model vars
        self.tcn = TemporalConvNet(num_inputs=self.input_size, num_channels=[self.hidden_size] * 3, kernel_size=2)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # load model parameters
        if not is_training:
            assert self.filename is not None, \
                "ERROR: spatial_ext_tcn.py: filename must be defined when is_training is False"
            self.load_model(self.filename)
        else:
            print("SpatialExtTCN is training")

    # Defining the forward pass
    def forward(self, x):
        # expects [batch_size, frames, features]
        #print("spatial x.shape0:", x.shape)
        batch_size = x.shape[0]
        '''
        if self.dense_data:
            print("spatial x.shape1:", x.shape)
            #x = x.view(1, 8, 512, -1)
            x = x.view(1, self.lfd_params.args.dense_rate, -1, self.input_size)

            x = x.mean(dim=3, keepdim=True)  # max consensus
            x = x.squeeze(3)
            x, _ = x.max(dim=1, keepdim=True)  # max consensus
            x = x.squeeze(1)
            print("spatial x.shape2:", x.shape)

        #x = x.view(self.lfd_params.args.batch_size, -1, self.input_size)
        
        else:
            if self.consensus == "max":
                x, _ = x.max(dim=1, keepdim=True)  # max consensus
                x = x.squeeze(1)
                x = torch.reshape(x, (batch_size, -1, self.input_size))  # ?
            elif self.consensus == "avg":
                x = x.mean(dim=1, keepdim=True)  # max consensus
                x = x.squeeze(1)
                x = torch.reshape(x, (batch_size, -1, self.input_size))  # ?
            elif self.consensus == "flat":
                x = torch.flatten(x, 1, 2)  # max consensus
        '''

        # combine visual features with empty action
        #print("spatial x.shape3:", x.shape)

        #h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        #c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

        # obtain logits
        x = torch.reshape(x, (batch_size, self.input_size, -1))
        print("x in shape:", x.shape)
        x = self.tcn(x)
        print("x out shape:", x.shape)
        #x, (h_out, _) = self.tcn(x, (h_0.detach(), c_0.detach()))
        x = x[:, :, -1]
        print("x in shape2:", x.shape)
        x = self.fc(x)
        print("x out shape2:", x.shape)

        #print("spatial x.shape4:", x.shape)

        if self.reshape_output:
            x = torch.squeeze(x, 1)

        return x

    def save_model(self):#, filename):
        torch.save(self.state_dict(), self.filename)
        print("SpatialExtTCN model saved to: ", self.filename)


    def load_model(self, filename):
        assert os.path.exists(filename), "ERROR: spatial_ext_tcn.py: Cannot locate saved model - "+filename

        print("Loading SpatialTCN from: " + filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint, strict=True)
        for param in self.parameters():
            param.requires_grad = False
