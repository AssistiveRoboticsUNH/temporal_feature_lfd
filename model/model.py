import torch
import torch.nn as nn
from torch.autograd import Variable


class LfDNetwork(nn.Module):
    def __init__(self, lfd_params, is_training=False):
        super().__init__()
        self.lfd_params = lfd_params

        self.use_ditrl = lfd_params.args.use_ditrl
        self.trim_model = lfd_params.args.trim_model

        # Observation feature extractor
        # --------
        if self.use_ditrl:
            from .temporal_feature_extractor import TemporalFeatureExtractor as FeatureExtractor
            self.observation_extractor = FeatureExtractor(lfd_params, train_model=is_training, use_pipeline=False)
        else:
            from .spatial_feature_extractor import SpatialFeatureExtractor as FeatureExtractor
            self.observation_extractor = FeatureExtractor(lfd_params, is_training)

        # Policy Generator
        # --------
        self.input_size = 11#self.lfd_params.num_actions + self.lfd_params.num_observations
        self.hidden_size = 5
        self.num_layers = 1
        self.num_actions = 4
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_actions)

        #checkpoint_file = lfd_params.args.policy_modelname

        filename = self.lfd_params.generate_policy_modelname()
        lstm_checkpoint_file = filename[:-3] + "lstm.pt"
        fc_checkpoint_file = filename[:-3] + "fc.pt"
        #if checkpoint_file:
        if lstm_checkpoint_file:
            '''
            print("Loading Policy Model from: "+checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            self.policy_output.load_state_dict(checkpoint, strict=True)
            for param in self.policy_output.parameters():
                param.requires_grad = False
            '''
            print("Loading LSTM Model from: " + lstm_checkpoint_file)
            checkpoint = torch.load(lstm_checkpoint_file)
            self.lstm.load_state_dict(checkpoint, strict=True)
            for param in self.lstm.parameters():
                param.requires_grad = False

            print("Loading FC Model from: " + fc_checkpoint_file)
            checkpoint = torch.load(fc_checkpoint_file)
            self.fc.load_state_dict(checkpoint, strict=True)
            for param in self.fc.parameters():
                param.requires_grad = False

        else:
            print("Did Not Load Policy Model")

    # Defining the forward pass
    def forward(self, obs_x, history):

        #print("obs_x:", obs_x.shape)
        #print("history:", history.shape)

        # extract visual features from observation
        obs_y = self.observation_extractor(obs_x.float())

        if self.trim_model:
            return obs_y

        #print("T1:", obs_y.shape)
        history[-1] = 0

        history = history.squeeze(axis=0)

        # combine visual features with empty action
        #state_x = state_x.type(torch.FloatTensor).view([-1, self.lfd_params.num_actions]).cuda()
        #action_x = Variable(torch.zeros(obs_y.shape[0], self.lfd_params.num_actions)).cuda()
        state_x = torch.cat([obs_y, history], dim=1, out=None)
        state_x = torch.unsqueeze(state_x, 0)
        #print("T2:", state_x.shape)

        # combine input history with most recent observation
        #state_x = torch.cat([history, state_x], dim=1, out=None)
        #print("T3:", state_x.shape)

        # create empty vars for LSTM
        h_0 = Variable(torch.zeros(self.num_layers, state_x.size(0), self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(self.num_layers, state_x.size(0), self.hidden_size)).cuda()
        #print("T3:", h_0.shape)

        # obtain logits
        state_y, (h_out, _) = self.lstm(state_x, (h_0.detach(), c_0.detach()))
        state_y = self.fc(state_y)
        state_y = state_y[:, -1, :]

        print("state_y.shape:", state_y.shape)
        print("state_x.shape:", state_x.shape)

        return state_y, state_x  # return the logits, and the input used

    def save_model(self, debug=False):
        self.observation_extractor.save_model()

        if debug:
            print("policy.state_dict():")
            for k in self.policy_output.state_dict().keys():
                print("\t"+k, self.policy_output.state_dict()[k].shape)

        filename = self.lfd_params.generate_policy_modelname()
        torch.save(self.lstm.state_dict(), filename[:-3]+"lstm.pt")
        torch.save(self.fc.state_dict(), filename[:-3]+"fc.pt")
        print("Policy model saved to: ", filename)
