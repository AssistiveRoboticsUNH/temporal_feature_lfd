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
        self.lstm = nn.LSTM(input_size=self.input_size + 7, hidden_size=self.hidden_size,
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

        batch_size = x.shape[1]
        x = torch.reshape(x, [batch_size, self.input_size, self.input_size, 7])

        x = x.detach().cpu().numpy()
        #print("x.shape:", x.shape)
        layered_x = []
        max_len = 0
        for i in range(batch_size):
            non_zero_idx = np.stack(np.nonzero(x[i])).T
            #print("len(non_zero_idx):", len(non_zero_idx))

            #print("non_zero_idx.shape:", non_zero_idx.shape)

            #input is matrix of shape (input x input x itrs(7))
            #non_zero_idx = torch.nonzero(x).detach().cpu().numpy()

            new_x = np.zeros((self.input_size+7, len(non_zero_idx)), np.float64)
            #print("non_zero_idx:", non_zero_idx)
            for idx in non_zero_idx:
                #print("idx:", idx, x[i, idx[0], idx[1], idx[2]])
                new_x[idx[0]] = 1
                new_x[idx[1]] = 1
                new_x[self.input_size + idx[2]] = x[i, idx[0], idx[1], idx[2]]
            #assert False, "stop here"

            layered_x.append(new_x)
            if new_x.shape[1] > max_len:
                max_len = new_x.shape[1]
            #print("new_x.shape", new_x.shape)

        for i in range(batch_size):
            layered_x[i] = np.pad(layered_x[i], ((0, 0), (0, max_len - layered_x[i].shape[1])), 'constant',
                                  constant_values=(0, 0))
            #print("layered_x[i]", layered_x[i].shape)
        layered_x = np.stack(layered_x)
        #print("layered_x.shape", layered_x.shape, layered_x.dtype)
        layered_x = np.transpose(layered_x, [0, 2, 1])


        x = torch.as_tensor(layered_x).cuda().float()


        #x = torch.reshape(x, (-1, self.input_size))

        #want an input of (input_size + itrs(7) x number of itrs). Each slice has  features


        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

        x, (h_out, _) = self.lstm(x, (h_0.detach(), c_0.detach()))
        x = self.fc(x)
        x = x[:, -1, :]

        return x

    def save_model(self, _):
            torch.save(self.lstm.state_dict(), self.lstm_filename)
            print("TemporalLSTM LSTM model saved to: ", self.lstm_filename)

            torch.save(self.fc.state_dict(), self.fc_filename)
            print("TemporalLSTM Linear model saved to: ", self.fc_filename)

    def load_model(self, filename, var):
        assert os.path.exists(filename), "ERROR: temporal_ext_linear.py: Cannot locate saved model - "+filename

        print("Loading TemporalExtLinear from: " + filename)
        checkpoint = torch.load(filename)
        var.load_state_dict(checkpoint, strict=True)
        for param in var.parameters():
            param.requires_grad = False
