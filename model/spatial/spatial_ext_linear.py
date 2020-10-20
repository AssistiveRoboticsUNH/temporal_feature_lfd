import os
import torch
import torch.nn as nn

# takes the output of a bottleneck filter and uses a max consensus and linear layer on the resultant features.


class SpatialExtLinear(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 input_size=128, output_size=8, consensus=None):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename
        self.consensus = consensus
        assert self.consensus in [None, "max", "avg"], \
            "ERROR: spatial_ext_linear.py: consensus must be either None, 'max', or 'avg'"

        # constants params
        self.input_size = input_size
        self.output_size = output_size

        # define model vars
        self.fc = nn.Sequential(
            nn.Linear(self.input_size, self.output_size),
            nn.Tanh()
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

        # print("spatial x.shape1:", x.shape)

        x = x.view((-1, self.lfd_params.args.num_segments) + x.size()[1:])
        # print("spatial x.shape2:", x.shape)
        x, _ = x.max(dim=1, keepdim=True)  # max consensus
        x = x.squeeze(1)
        # print("spatial x.shape3:", x.shape)

        x = torch.reshape(x, (-1, self.input_size))

        # print("spatial x.shape4:", x.shape)
        return self.fc(x)

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
