import os
import torch
import torch.nn as nn

from torch.autograd import Variable
import numpy as np


class TemporalExtLSTM(nn.Module):
    def __init__(self, lfd_params, is_training=False, filename=None,
                 input_size=11, output_size=4):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename
        self.lstm_filename = ".".join([self.filename, "temporal_lstm", "pt"])
        self.fc_filename = ".".join([self.filename, "temporal_fc", "pt"])

        # constants params
        self.input_size = input_size
        self.hidden_size = 32
        self.num_layers = 1
        self.output_size = output_size

        # define model vars
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # load model parameters
        if not is_training:
            assert self.filename is not None, \
                "ERROR: temporal_ext_lstm.py: filename must be defined when is_training is False"
            self.load_model(self.lstm_filename, self.lstm)
            self.load_model(self.fc_filename, self.fc)
        else:
            print("SpatialExtLSTM is training")

    # Defining the forward pass
    def forward(self, x):

        x = x.detach().cpu().numpy()
        print("x.shape:", x.shape)

        non_zero_idx = np.nonzero(x)
        print("non_zero_idx.shape:", non_zero_idx.shape)

        #input is matrix of shape (input x input x itrs(7))
        #non_zero_idx = torch.nonzero(x).detach().cpu().numpy()
        
        new_x = np.zeros((self.input_size+7, len(non_zero_idx)))
        print("non_zero_idx:", non_zero_idx)
        for idx in non_zero_idx:
            new_x[idx[0]] = 1
            new_x[idx[1]] = 1
            new_x[self.input_size + idx[2]] = x[non_zero_idx]
        assert False, "stop here"

        x = torch.as_tensor(new_x)

        #x = torch.reshape(x, (-1, self.input_size))

        #want an input of (input_size + itrs(7) x number of itrs). Each slice has  features

        '''
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

        x, (h_out, _) = self.lstm(x, (h_0.detach(), c_0.detach()))
        x = self.fc(x)
        x = x[:, -1, :]
        '''
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
