import os
import torch.nn as nn

from .backbone_model.backbone_i3d import BackboneI3D
from .spatial.spatial_bottleneck import SpatialBottleneck
from .spatial.spatial_ext_linear import SpatialExtLinear
from .temporal.temporal_pipeline import TemporalPipeline
from .temporal.temporal_ext_linear import TemporalExtLinear


class ClassifierDITRLI3D(nn.Module):
    def __init__(self, lfd_params, filename,
                 use_feature_extractor=False,
                 spatial_train=False, use_spatial=False,
                 ditrl_pipeline_train=False, use_pipeline=False,
                 temporal_train=False, use_temporal=False):
        super().__init__()
        self.lfd_params = lfd_params
        self.backbone_id = "i3d"

        # parts of model to train
        self.spatial_train = spatial_train
        self.ditrl_pipeline_train = ditrl_pipeline_train
        self.temporal_train = temporal_train

        self.use_feature_extractor = use_feature_extractor  # use to get features for IAD
        self.use_spatial = use_spatial  # use to get classification from IAD
        self.use_pipeline = use_pipeline  # use to get ITRs from IAD
        self.use_temporal = use_temporal  #use to learn from ITRs

        # model filenames
        self.filename = filename

        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.bottleneck_filename = ".".join([self.filename, "bottleneck", "pt"])
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])
        self.pipeline_filename = ".".join([self.filename, "pipeline", "pk"])
        self.temporal_filename = ".".join([self.filename, "temporal", "pt"])

        # model sections
        if use_feature_extractor:
            pretrain_modelname = os.path.join(lfd_params.args.home_dir,
                                              "models/rgb_imagenet.pt")
            self.backbone = BackboneI3D(lfd_params, is_training=self.spatial_train, trim_model=True,
                                        filename=pretrain_modelname if spatial_train else self.backbone_filename)
            self.bottleneck = SpatialBottleneck(lfd_params, is_training=self.spatial_train,
                                                input_size=1024,
                                                filename=self.bottleneck_filename,
                                                bottleneck_size=lfd_params.args.bottleneck_size)
        if use_spatial:
            self.spatial = SpatialExtLinear(lfd_params, is_training=self.spatial_train,
                                            filename=self.spatial_filename,
                                            input_size=lfd_params.args.bottleneck_size,
                                            consensus="max", dense_data=True)
        if use_pipeline:
            self.pipeline = TemporalPipeline(lfd_params, is_training=self.ditrl_pipeline_train,
                                             filename=self.pipeline_filename)
        if use_temporal:
            self.temporal = TemporalExtLinear(lfd_params, is_training=self.temporal_train,
                                              filename=self.temporal_filename,
                                              input_size=(lfd_params.args.bottleneck_size**2 * 7),
                                              output_size=8)

    # Defining the forward pass
    def forward(self, x):
        print("classifier_ditrl_tsm.py: input:", x.shape)


        if self.use_feature_extractor:
            x = self.backbone(x)
            x = self.bottleneck(x)
            print("classifier_ditrl_tsm.py: use_feature_extractor:", x.shape)
        if self.use_spatial:
            x = self.spatial(x)
            print("classifier_ditrl_tsm.py: spatial:", x.shape)
        if self.use_pipeline:
            x = self.pipeline(x)
            print("classifier_ditrl_tsm.py: pipeline:", x.shape)
        if self.use_temporal:
            x = self.temporal(x)
            print("classifier_ditrl_tsm.py: temporal:", x.shape)
        return x

    def save_model(self):
        if self.use_feature_extractor and self.spatial_train:
            self.backbone.save_model(self.backbone_filename)
            self.bottleneck.save_model(self.bottleneck_filename)
        if self.use_spatial and self.spatial_train:
            self.spatial.save_model(self.spatial_filename)
        if self.use_pipeline and self.ditrl_pipeline_train:
            self.pipeline.save_model(self.pipeline_filename)
        if self.use_temporal and self.temporal_train:
            self.temporal.save_model(self.temporal_filename)
