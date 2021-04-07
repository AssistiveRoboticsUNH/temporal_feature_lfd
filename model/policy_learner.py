import torch

from .classifier import Classifier
from .policy.policy_lstm import PolicyLSTM
import torch.nn as nn
import os


class PolicyLearner(Classifier):
    def __init__(self, lfd_params, filename, backbone_id, suffix,
                 use_backbone=False, train_backbone=False,
                 use_bottleneck=False, train_bottleneck=False,
                 use_spatial=False, train_spatial=False,
                 use_pipeline=False, train_pipeline=False,
                 use_temporal=False, train_temporal=False,
                 train_policy=False):
        super().__init__(lfd_params, filename, backbone_id, suffix,
                         use_backbone=use_backbone, train_backbone=train_backbone,
                         use_bottleneck=use_bottleneck, train_bottleneck=train_bottleneck,
                         use_spatial=use_spatial, train_spatial=train_spatial,
                         use_pipeline=use_pipeline, train_pipeline=train_pipeline,
                         use_temporal=use_temporal, train_temporal=train_temporal,
                         policy_learn_ext=True)

        # parts of model to train
        self.train_policy = train_policy

        # model filenames
        #self.filename = os.path.join(self.lfd_params.model_save_dir, filename) # taken care of by classifer
        #self.lstm_filename = ".".join([self.filename, "lstm", "pt"])
        #self.fc_filename = ".".join([self.filename, "policy", "pt"])

        # model sections
        input_size = len(lfd_params.application.obs_label_list) + len(lfd_params.application.act_label_list)
        self.policy = PolicyLSTM(self.lfd_params, is_training=self.train_policy,
                                 input_size=input_size,
                                 filename=self.filename)
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
