import torch

from .classifier_ditrl import ClassifierDITRL
from .policy.policy_lstm import PolicyLSTM
import torch.nn as nn
import numpy as np

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
        self.activation = nn.Softmax(dim=2)
        self.policy = PolicyLSTM(lfd_params, is_training=policy_train, input_size=8+4,
                                 lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)

    # Defining the forward pass
    def forward(self, obs_x, act_x):
        obs_x = super().forward(obs_x)
        obs_x = torch.unsqueeze(obs_x, 0)

        #print("obs_x:", obs_x)
        obs_x = self.activation(obs_x)

        print("obs_x:", obs_x)
        idx = torch.argmax(obs_x, dim=2).detach().cpu().numpy()[0]
        print("idx:", idx)

        new_obs = np.zeros_like(obs_x.detach().cpu().numpy())
        print("new_obs:", new_obs.shape)
        for r in idx.shape:
            new_obs[0, r, idx[r]] = 1
        print("new_obs2:", new_obs)

        obs_x = torch.as_tensor(new_obs).cuda()

        #print("obs_x_soft:", obs_x)
        x = self.policy(obs_x, act_x)
        return x

    def save_model(self):
        super().save_model()
        if self.policy_train:
            self.policy.save_model(lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)
