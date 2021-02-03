import torch

from .classifier import Classifier
from .policy.policy_lstm import PolicyLSTM
import torch.nn as nn


class PolicyLearner(Classifier):
    def __init__(self, lfd_params, filename, backbone_id, suffix, 
                 use_feature_extractor=False, train_feature_extractor=False,
                 use_spatial=False, train_spatial=False,
                 use_pipeline=False, train_pipeline=False,
                 use_temporal=False, train_temporal=False,
                 train_policy=False):
        super().__init__(lfd_params, filename, backbone_id, suffix, 
                         use_feature_extractor=use_feature_extractor, train_feature_extractor=train_feature_extractor,
                         use_spatial=use_spatial, train_spatial=train_spatial,
                         use_pipeline=use_pipeline, train_pipeline=train_pipeline,
                         use_temporal=use_temporal, train_temporal=train_temporal,
                         policy_learn_ext=True)

        # parts of model to train
        self.train_policy = train_policy

        # model filenames
        #self.lstm_filename = ".".join([self.filename, "lstm", "pt"])
        #self.fc_filename = ".".join([self.filename, "policy", "pt"])

        # model sections
        self.policy = PolicyLSTM(lfd_params, is_training=train_policy, input_size=8+4, filename=filename)
                                 #lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)

    # Defining the forward pass
    def forward(self, obs_x, act_x):
        obs_x = super().forward(obs_x)
        obs_x = torch.unsqueeze(obs_x, 0)

        x = self.policy(obs_x, act_x)
        return x

    def save_model(self):
        print("save called on policy_learner.py, train_policy:", self.train_policy)
        super().save_model()
        if self.train_policy:
            print("trying to save policy")
            self.policy.save_model()#lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)
