import os
import torch
import torch.nn as nn

from torch.autograd import Variable


class PolicyLSTM(nn.Module):
    def __init__(self, lfd_params, is_training=False, lstm_filename=None, fc_filename=None,
                 input_size=11, hidden_size=5, num_layers=1, output_size=4):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.lstm_filename = lstm_filename
        self.fc_filename = fc_filename

        # constants params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # define model vars
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # load model parameters
        if not is_training:
            assert self.lstm_filename is not None and self.fc_filename is not None, \
                "ERROR: policy_lstm.py: lstm_filename AND fc_filename must be defined when is_training is False"
            self.load_model(self.lstm_filename, self.lstm)
            self.load_model(self.fc_filename, self.fc)
        else:
            print("PolicyLSTM is training")

    # Defining the forward pass
    def forward(self, obs_x, act_x):

        # combine visual features with empty action
        state_x = torch.cat([obs_x, act_x], dim=1, out=None)
        state_x = torch.unsqueeze(state_x, 0)

        print("lstm policy state_x.shape:", state_x.shape)

        # create empty vars for LSTM
        h_0 = Variable(torch.zeros(self.num_layers, state_x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(self.num_layers, state_x.size(0), self.hidden_size)).cuda()

        # obtain logits
        state_y, (h_out, _) = self.lstm(state_x, (h_0.detach(), c_0.detach()))
        state_y = self.fc(state_y)
        state_y = state_y[:, -1, :]

        # return the action logits
        return state_y

    def save_model(self, lstm_filename, fc_filename):
        torch.save(self.lstm.state_dict(), lstm_filename)
        print("PolicyLSTM LSTM model saved to: ", lstm_filename)

        torch.save(self.fc.state_dict(), fc_filename)
        print("PolicyLSTM Linear model saved to: ", fc_filename)

    def load_model(self, filename, var):
        assert os.path.exists(filename), "ERROR: policy_lstm.py: Cannot locate saved model - "+filename

        print("Loading PolicyLSTM from: " + filename)
        checkpoint = torch.load(filename)
        var.load_state_dict(checkpoint, strict=True)
        for param in var.parameters():
            param.requires_grad = False