import os
import torch
import torch.nn as nn

# takes the output of a bottleneck filter and uses a max consensus and linear layer on the resultant features.


class SpatialExtLinear(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 input_size=128, output_size=7):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename

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

        x = x.view((-1, self.rgb_net.num_segments) + x.size()[1:])
        x, _ = x.max(dim=1, keepdim=True)  # max consensus
        x = x.squeeze(1)

        x = torch.reshape(x, (-1, self.input_size))
        return self.fc(x)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        print("SpatialExtLinear Linear model saved to: ", filename)

    def load_model(self, filename, var):
        assert os.path.exists(filename), "ERROR: spatial_ext_linear.py: Cannot locate saved model - " + filename

        print("Loading SpatialExtLinear from: " + filename)
        checkpoint = torch.load(filename)
        var.load_state_dict(checkpoint, strict=True)
        for param in var.parameters():
            param.requires_grad = False
