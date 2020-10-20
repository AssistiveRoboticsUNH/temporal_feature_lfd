import os
import torch
import torch.nn as nn


class TemporalExtLinear(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 input_size=11, output_size=4):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename

        # constants params
        self.input_size = input_size
        self.output_size = output_size

        # define model vars
        self.fc = nn.Linear(self.input_size, self.output_size)

        # load model parameters
        if not is_training:
            assert self.filename is not None, \
                "ERROR: temporal_ext_linear.py: filename must be defined when is_training is False"
            self.load_model(self.filename, self.fc)
        else:
            print("TemporalExtLinear is training")

    # Defining the forward pass
    def forward(self, x):

        print("temporal_ext_linear.py 1: ", x.shape)
        x = torch.reshape(x, (-1, self.input_size))
        print("temporal_ext_linear.py 2: ", x.shape)
        x = self.fc(x)
        x = torch.unsqueeze(x, 0)
        print("temporal_ext_linear.py 3: ", x.shape)
        return x

    def save_model(self, filename):
        torch.save(self.fc.state_dict(), filename)
        print("TemporalExtLinear Linear model saved to: ", filename)

    def load_model(self, filename, var):
        assert os.path.exists(filename), "ERROR: temporal_ext_linear.py: Cannot locate saved model - "+filename

        print("Loading TemporalExtLinear from: " + filename)
        checkpoint = torch.load(filename)
        var.load_state_dict(checkpoint, strict=True)
        for param in var.parameters():
            param.requires_grad = False
