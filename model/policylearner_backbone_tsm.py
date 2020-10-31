import os
import torch
import torch.nn as nn

from .backbone_model.backbone_tsm import BackboneTSM
from .spatial.spatial_ext_linear import SpatialExtLinear
from .policy.policy_lstm import PolicyLSTM


class PolicyLearnerBackboneTSM(nn.Module):
    def __init__(self, lfd_params, filename,
                 spatial_train=False,
                 policy_train=False):
        super().__init__()
        self.lfd_params = lfd_params
        self.backbone_id = "tsm"

        # model filenames
        self.filename = filename

        # parts of model to train
        self.spatial_train = spatial_train
        self.policy_train = policy_train

        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])
        self.lstm_filename = ".".join([self.filename, "lstm", "pt"])
        self.fc_filename = ".".join([self.filename, "policy", "pt"])

        # model sections
        pretrain_modelname = os.path.join(lfd_params.args.home_dir,
                                          "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
        self.backbone = BackboneTSM(lfd_params, is_training=spatial_train,
                                    filename=pretrain_modelname if spatial_train else self.backbone_filename)
        self.spatial = SpatialExtLinear(lfd_params, is_training=spatial_train, filename=self.spatial_filename,
                                        input_size=2048, consensus="max")
        self.policy = PolicyLSTM(lfd_params, is_training=policy_train,
                                 lstm_filename=self.lstm_filename, fc_filename=self.fc_filename,
                                 )

        self.model = nn.Sequential(
            self.backbone,
            self.spatial)

    # Defining the forward pass
    def forward(self, obs_x, act_x):
        obs_y = self.backbone(obs_x)
        print("obs_y1:", obs_y.shape)
        obs_y = self.spatial(obs_x)
        print("obs_y2:", obs_y.shape)


        obs_y = torch.unsqueeze(obs_y, 0)

        print("obs_y3:", obs_y.shape)
        print("act_x:", act_x.shape)

        return self.policy(obs_y, act_x)

    def save_model(self):
        if self.spatial_train:
            self.backbone.save_model(self.backbone_filename)
            self.spatial.save_model(self.spatial_filename)
        if self.policy_train:
            self.policy.save_model(lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)
