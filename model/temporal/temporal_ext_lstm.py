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
        self.filename = os.path.join(filename, ".".join(["model", "temporal_lstm", "pt"]))

        # constants params
        self.input_size = input_size
        self.hidden_size = 16
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
            self.load_model(self.filename)
        else:
            print("SpatialExtLSTM is training")

    # Defining the forward pass
    def forward(self, x):
        x, edge_idx, edge_attr, batch = x.x, x.edge_index, x.edge_attr, x.batch

        x = x.cpu().numpy()
        edge_idx = edge_idx.cpu().numpy()
        edge_attr = edge_attr.cpu().numpy()
        batch = batch.cpu().numpy()
        batch_size = len(np.unique(batch))

        layered_x = []
        max_len = 0

        for i in range(batch_size):
            new_x = np.zeros((max(1, edge_idx.shape[1]), self.input_size + 7), np.float64)
            edge_idxes = np.where(batch == i)[0]

            for j in edge_idxes:
                n1 = edge_idx[0, j]
                n2 = edge_idx[1, j]
                itr = edge_attr[j]

                n1_value = x[n1]
                n2_value = x[n2]

                node_value = n1_value + n2_value
                itr_value = np.zeros(7)
                itr_value[itr] = 1

                new_x[j - edge_idxes[0]] = np.concatenate((node_value, itr_value))

            layered_x.append(new_x)
            if new_x.shape[1] > max_len:
                max_len = new_x.shape[1]

        for i in range(batch_size):
            layered_x[i] = layered_x[i][:max_len]
        layered_x = np.stack(layered_x)

        x = torch.as_tensor(layered_x).cuda().float()

        #want an input of (input_size + itrs(7) x number of itrs). Each slice has  features
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()

        x, (h_out, _) = self.lstm(x, (h_0.detach(), c_0.detach()))
        x = self.fc(x)
        x = x[:, -1, :]

        return x

    def save_model(self):
        torch.save(self.state_dict(), self.filename)
        print("TemporalExtLSTM model saved to: ", self.filename)

    def load_model(self, filename):
        assert os.path.exists(filename), "ERROR: temporal_ext_linear.py: Cannot locate saved model - " + filename

        print("Loading TemporalExtLSTM from: " + filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint, strict=True)
        for param in self.parameters():
            param.requires_grad = False


