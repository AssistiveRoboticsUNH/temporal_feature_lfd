import torch.nn as nn
import os
from enums import model_dict, Backbone

from .spatial.spatial_bottleneck import SpatialBottleneck


class FeatureExtractor(nn.Module):
    def __init__(self, lfd_params, filename, backbone_id,
                 use_backbone=True, backbone_train=False,
                 use_bottleneck=False, bottleneck_train=False):
        super().__init__()
        self.lfd_params = lfd_params
        self.backbone_id = backbone_id

        # parts of model to train
        self.backbone_train = backbone_train
        self.bottleneck_train = bottleneck_train

        self.use_backbone = use_backbone
        self.use_bottleneck = use_bottleneck  # use to get features for IAD

        # model filenames
        self.filename = filename
        self.backbone_filename = os.path.join(filename, ".".join(["model", "backbone", "pt"]))
        #self.bottleneck_filename = os.path.join(filename, ".".join(["model", "spatial_bottleneck", "pt"]))

        # model sections
        backbone_class = self.lfd_params.model.backbone_class
        pretrain_model_name = self.lfd_params.model.pretrain_model_name

        print("backbone_class:", backbone_class)

        self.num_output_features = lfd_params.model.original_size
        if self.use_backbone:
            self.backbone = backbone_class(self.lfd_params,
                                           is_training=self.backbone_train,
                                           trim_model=use_bottleneck,
                                           filename=pretrain_model_name,# if self.backbone_train else self.backbone_filename,
                                           end_point=lfd_params.model.end_point)

        if self.use_bottleneck:
            self.bottleneck = SpatialBottleneck(self.lfd_params,
                                                is_training=self.bottleneck_train,
                                                filename=self.filename,
                                                bottleneck_size=self.lfd_params.model.bottleneck_size,
                                                input_size=self.lfd_params.model.original_size,
                                                spatial_size=self.lfd_params.model.spatial_size)
            self.num_output_features = self.lfd_params.model.bottleneck_size

    # Defining the forward pass
    def forward(self, x):
        if self.use_backbone:
            x = self.backbone(x)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        return x

    def save_model(self):
        if self.use_backbone and self.backbone_train:
            # we need to pass the filename parameter to make sure we don't overwrite the pretrained model
            self.backbone.save_model(self.backbone_filename)
        if self.use_bottleneck and self.bottleneck_train:
            self.bottleneck.save_model()#self.bottleneck_filename)
