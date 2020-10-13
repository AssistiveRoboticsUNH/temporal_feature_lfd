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
        self.input_size = self.lfd_params.num_actions + self.lfd_params.num_observations
        self.hidden_size = 5
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.lfd_params.num_actions)

        checkpoint_file = lfd_params.args.policy_modelname
        if checkpoint_file:
            print("Loading Policy Model from: "+checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            self.policy_output.load_state_dict(checkpoint, strict=True)
            for param in self.policy_output.parameters():
                param.requires_grad = False
        else:
            print("Did Not Load Policy Model")

    # Defining the forward pass
    def forward(self, obs_x, history):

        print("obs_x:", obs_x.shape)
        print("history:", history.shape)

        # extract visual features from observation
        obs_y = self.observation_extractor(obs_x.float())

        if self.trim_model:
            return obs_y

        print("T1:", obs_y.shape)
        history[-1] = 0

        print("Q1", obs_y.dtype)
        print("Q2:", history.dtype)

        # combine visual features with empty action
        #state_x = state_x.type(torch.FloatTensor).view([-1, self.lfd_params.num_actions]).cuda()
        #action_x = Variable(torch.zeros(obs_y.shape[0], self.lfd_params.num_actions)).cuda()
        state_x = torch.stack([obs_y, history])
        print("T2:", state_x.shape)

        # combine input history with most recent observation
        #state_x = torch.cat([history, state_x], dim=1, out=None)
        #print("T3:", state_x.shape)

        # create empty vars for LSTM
        h_0 = Variable(torch.zeros(self.num_layers, state_x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, state_x.size(0), self.hidden_size))

        # obtain logits
        state_y, (h_out, _) = self.lstm(state_x, (h_0.detach(), c_0.detach()))
        state_y = self.fc(state_y)
        state_y = state_y[:, -1, :]

        return state_y, state_x  # return the logits, and the input used

    def save_model(self, debug=False):
        self.observation_extractor.save_model(debug)

        if debug:
            print("policy.state_dict():")
            for k in self.policy_output.state_dict().keys():
                print("\t"+k, self.policy_output.state_dict()[k].shape )

        filename = self.lfd_params.generate_policy_modelname()
        torch.save(self.policy_output.state_dict(), filename )
        print("Policy model saved to: ", filename)
