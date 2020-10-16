import os
import torch
import torch.nn as nn

from .backbone_model.backbone_tsm import BackboneTSM
from .spatial.spatial_bottleneck import SpatialBottleneck
from .spatial.spatial_ext_linear import SpatialExtLinear
from .temporal.temporal_pipeline import TemporalPipeline
from .temporal.temporal_ext_linear import TemporalExtLinear
from .policy.policy_lstm import PolicyLSTM


class ModularModel(nn.Module):
    def __init__(self, lfd_params,
                 train_spatial=False,
                 train_pipeline=False,
                 train_policy=False):
        super().__init__()
        self.lfd_params = lfd_params

        # model filenames
        self.filename = filename

        # model sections
        self.backbone = BackboneTSM(lfd_params, is_training=False)
        self.spatial = nn.Sequential(SpatialBottleneck(lfd_params, is_training=False),
                                     SpatialExtLinear(lfd_params, is_training=False)
                                     )
        self.temporal = nn.Sequential(TemporalPipeline(lfd_params, is_training=False),
                                      TemporalExtLinear(lfd_params, is_training=False),
                                     )
        self.policy = PolicyLSTM(lfd_params, is_training=False)





        self.model = nn.Sequential(
            self.backbone,
            self.spatial,
            self.temporal,
            self.policy
        )

    # Defining the forward pass
    def forward(self, x):
        return self.model(x)

    def save_model(self, filename):
        pass
