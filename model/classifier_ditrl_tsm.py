import torch.nn as nn

from .backbone_model.backbone_tsm import BackboneTSM
from .spatial.spatial_bottleneck import SpatialBottleneck
from .spatial.spatial_ext_linear import SpatialExtLinear
from .temporal.temporal_pipeline import TemporalPipeline
from .temporal.temporal_ext_linear import TemporalExtLinear


class ClassifierDITRLTSM(nn.Module):
    def __init__(self, lfd_params, filename,
                 spatial_train=False, use_spatial=True,
                 ditrl_pipeline_train=False, use_pipeline=True,
                 temporal_train=False, use_temporal=True):
        super().__init__()
        self.lfd_params = lfd_params

        # parts of model to train
        self.spatial_train = spatial_train
        self.ditrl_pipeline_train = ditrl_pipeline_train
        self.temporal_train = temporal_train

        self.use_spatial = use_spatial
        self.use_pipeline = use_pipeline
        self.use_temporal = use_temporal

        # model filenames
        self.filename = filename

        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.bottleneck_filename = ".".join([self.filename, "bottleneck", "pt"])
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])
        self.pipeline_filename = ".".join([self.filename, "pipeline", "pk"])
        self.temporal_filename = ".".join([self.filename, "temporal", "pt"])

        # model sections
        if use_spatial:
            self.backbone = BackboneTSM(lfd_params, is_training=self.spatial_train,
                                        filename=lfd_params.args.pretrain_modelname if spatial_train else self.backbone)
            self.bottleneck = SpatialBottleneck(lfd_params, is_training=self.spatial_train,
                                                filename=self.bottleneck_filename,
                                                bottleneck_size=lfd_params.args.bottleneck_size)
            self.spatial = SpatialExtLinear(lfd_params, is_training=self.spatial_train,
                                            filename=self.spatial_filename,
                                            input_size=lfd_params.args.bottleneck_size)
            self.feature_extractor = nn.Sequential(
                self.backbone,
                self.bottleneck,
                self.spatial,
            )

        if use_pipeline:
            self.pipeline = TemporalPipeline(lfd_params, is_training=self.ditrl_pipeline_train,
                                             filename=self.pipeline_filename)
        if use_temporal:
            self.temporal = TemporalExtLinear(lfd_params, is_training=self.temporal_train,
                                              filename=self.temporal_filename)

    # Defining the forward pass
    def forward(self, x):

        if self.use_spatial:
            x = self.feature_extractor(x)
        if self.use_pipeline:
            x = self.pipeline(x)
        if self.use_temporal:
            x = self.temporal(x)
        return x

    def save_model(self):
        if self.spatial_train:
            self.backbone.save_model(self.backbone_filename)
            self.bottleneck.save_model(self.bottleneck_filename)
            self.spatial.save_model(self.spatial_filename)
        if self.ditrl_pipeline_train:
            self.pipeline.save_model(self.pipeline_filename)
        if self.temporal_train:
            self.temporal.save_model(self.temporal_filename)
