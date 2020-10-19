import torch.nn as nn

from .backbone_model.backbone_tsm import BackboneTSM
from .spatial.spatial_bottleneck import SpatialBottleneck
from .spatial.spatial_ext_linear import SpatialExtLinear
from .temporal.temporal_pipeline import TemporalPipeline
from .temporal.temporal_ext_linear import TemporalExtLinear


class ClassifierDITRLTSM(nn.Module):
    def __init__(self, lfd_params, filename,
                 spatial_train=False,
                 ditrl_pipeline_train=False,
                 temporal_train=False):
        super().__init__()
        self.lfd_params = lfd_params

        # parts of model to train
        self.spatial_train = spatial_train
        self.ditrl_pipeline_train = ditrl_pipeline_train
        self.temporal_train = temporal_train

        # model filenames
        self.filename = filename

        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.bottleneck_filename = ".".join([self.filename, "bottleneck", "pt"])
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])
        self.pipeline_filename = ".".join([self.filename, "pipeline", "pk"])
        self.temporal_filename = ".".join([self.filename, "temporal", "pt"])

        # model sections
        self.backbone = BackboneTSM(lfd_params, is_training=self.spatial_train, filename=self.backbone_filename)
        self.bottleneck = SpatialBottleneck(lfd_params, is_training=self.spatial_train, filename=self.bottleneck_filename)
        self.spatial = SpatialExtLinear(lfd_params, is_training=self.spatial_train, filename=self.spatial_filename)
        self.pipeline = TemporalPipeline(lfd_params, is_training=self.ditrl_pipeline_train, filename=self.pipeline_filename)
        self.temporal = TemporalExtLinear(lfd_params, is_training=self.temporal_train, filename=self.temporal_filename)

        self.feature_extractor = nn.Sequential(
            self.backbone,
            self.bottleneck,
            self.spatial,
        )

        self.model = nn.Sequential(
            self.feature_extractor,
            self.pipeline,
            self.temporal,
        )

    # Defining the forward pass
    def forward(self, x, trim_after_spatial=False, trim_after_pipeline=False):
        x = self.feature_extractor(x)
        if trim_after_spatial:
            return x

        x = self.pipeline(x)
        if trim_after_pipeline:
            return x

        return self.temporal(x)

    def save_model(self):
        if self.spatial_train:
            self.backbone.save_model(self.backbone_filename)
            self.bottleneck.save_model(self.bottleneck_filename)
            self.spatial.save_model(self.spatial_filename)
        if self.ditrl_pipeline_train:
            self.pipeline.save_model(self.pipeline_filename)
        if self.temporal_train:
            self.temporal.save_model(self.temporal_filename)
