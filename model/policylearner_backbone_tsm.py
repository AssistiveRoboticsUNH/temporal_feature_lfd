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

        # model filenames
        self.filename = filename

        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])
        self.lstm_filename = ".".join([self.filename, "lstm", "pt"])
        self.fc_filename = ".".join([self.filename, "policy", "pt"])

        # model sections
        self.backbone = BackboneTSM(lfd_params, is_training=spatial_train, filename=self.backbone_filename)
        self.spatial = SpatialExtLinear(lfd_params, is_training=spatial_train, filename=self.spatial_filename)
        self.policy = PolicyLSTM(lfd_params, is_training=policy_train, lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)

        self.model = nn.Sequential(
            self.backbone,
            self.spatial,
            self.policy
        )

    # Defining the forward pass
    def forward(self, x):
        return self.model(x)

    def save_model(self):
        if self.spatial_train:
            self.backbone.save_model(self.backbone_filename)
            self.spatial.save_model(self.spatial_filename)
        if self.policy_train:
            self.policy.save_model(lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)
