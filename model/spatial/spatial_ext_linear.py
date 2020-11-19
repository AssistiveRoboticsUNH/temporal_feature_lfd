import os
import torch
import torch.nn as nn

# takes the output of a bottleneck filter and uses a max consensus and linear layer on the resultant features.


class SpatialExtLinear(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 input_size=128, output_size=8, consensus=None, dense_data=False, reshape_output=False):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename
        self.consensus = consensus
        self.reshape_output = reshape_output


        assert self.consensus in [None, "max", "avg", "flat"], \
            "ERROR: spatial_ext_linear.py: consensus must be either None, 'max', 'avg', of 'flat'"
        self.dense_data = dense_data

        # constants params
        self.input_size = input_size
        self.output_size = output_size

        # define model vars
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            #nn.Tanh()
        )

        # load model parameters
        if not is_training:
            assert self.filename is not None, \
                "ERROR: spatial_ext_linear.py: filename must be defined when is_training is False"
            self.load_model(self.filename, self.fc)
        else:
            print("SpatialExtLinear is training")

    # Defining the forward pass
    def forward(self, x):
        # expects [batch_size, frames, features]
        print("spatial x.shape0:", x.shape)
        batch_size = x.shape[0]

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

        print("spatial x.shape3:", x.shape)
        x = self.fc(x)
        print("spatial x.shape4:", x.shape)

        if self.reshape_output:
            x = torch.squeeze(x, 1)

        return x

    def save_model(self, filename):
        torch.save(self.fc.state_dict(), filename)
        print("SpatialExtLinear Linear model saved to: ", filename)

    def load_model(self, filename, var):
        assert os.path.exists(filename), "ERROR: spatial_ext_linear.py: Cannot locate saved model - " + filename

        print("Loading SpatialExtLinear from: " + filename)
        checkpoint = torch.load(filename)

        print("checkpoint")
        for k in checkpoint.keys():
            print(k)

        var.load_state_dict(checkpoint, strict=True)
        for param in var.parameters():
            param.requires_grad = False
