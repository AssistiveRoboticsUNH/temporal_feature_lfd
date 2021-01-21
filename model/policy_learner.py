import torch

from .classifier import Classifier
from .policy.policy_lstm import PolicyLSTM
import torch.nn as nn


class PolicyLearner(Classifier):
    def __init__(self, lfd_params, filename, backbone_id,
                 use_feature_extractor=False, spatial_train=False, use_spatial=False, use_spatial_lstm=False,
                 policy_train=False, use_bottleneck=True):
        super().__init__(lfd_params, filename, backbone_id,
                         use_feature_extractor=use_feature_extractor,
                         spatial_train=spatial_train, use_spatial=use_spatial, use_spatial_lstm=use_spatial_lstm,
                         use_bottleneck=use_bottleneck, policy_learn_ext=True)

        # parts of model to train
        self.policy_train = policy_train

        # model filenames
        self.lstm_filename = ".".join([self.filename, "lstm", "pt"])
        self.fc_filename = ".".join([self.filename, "policy", "pt"])

        # model sections
        #self.activation = nn.Tanh()
        self.policy = PolicyLSTM(lfd_params, is_training=policy_train, input_size=8+4,
                                 lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)

    # Defining the forward pass
    def forward(self, obs_x, act_x):
        obs_x = super().forward(obs_x)
        obs_x = torch.unsqueeze(obs_x, 0)

        #print("obs_x:", obs_x.shape)
        #print("act_x:", act_x.shape)
        #obs_x = self.activation(obs_x)
        x = self.policy(obs_x, act_x)
        return x

    def save_model(self):
        super().save_model()
        if self.policy_train:
            self.policy.save_model(lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)
