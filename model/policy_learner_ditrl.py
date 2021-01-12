import torch

from .classifier_ditrl import ClassifierDITRL
from .policy.policy_lstm import PolicyLSTM
import torch.nn as nn


class PolicyLearnerDITRL(ClassifierDITRL):
    def __init__(self, lfd_params, filename, backbone_id,
                 use_feature_extractor=False,
                 spatial_train=False, use_spatial=True,
                 ditrl_pipeline_train=False, use_pipeline=False,
                 return_iad=False, return_vee=False,
                 temporal_train=False, use_temporal=False, use_gcn=False,
                 policy_train=False, use_itr_lstm=False):
        super().__init__(lfd_params, filename, backbone_id, use_feature_extractor, spatial_train, use_spatial,
                         ditrl_pipeline_train, use_pipeline,
                         return_iad, return_vee,
                         temporal_train, use_temporal, use_gcn, use_itr_lstm)

        # parts of model to train
        self.policy_train = policy_train

        # model filenames
        self.lstm_filename = ".".join([self.filename, "lstm", "pt"])
        self.fc_filename = ".".join([self.filename, "policy", "pt"])

        # model sections
        self.activation = nn.LogSoftmax()
        self.policy = PolicyLSTM(lfd_params, is_training=policy_train, input_size=8+4,
                                 lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)

    # Defining the forward pass
    def forward(self, obs_x, act_x):
        obs_x = super().forward(obs_x)
        obs_x = torch.unsqueeze(obs_x, 0)

        print("obs_x:", obs_x)
        obs_x = self.activation(obs_x)
        print("obs_x_soft:", obs_x)
        x = self.policy(obs_x, act_x)
        return x

    def save_model(self):
        super().save_model()
        if self.policy_train:
            self.policy.save_model(lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)
