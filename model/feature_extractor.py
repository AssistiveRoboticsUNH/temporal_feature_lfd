import torch.nn as nn
import os

from .spatial.spatial_bottleneck import SpatialBottleneck


class FeatureExtractor(nn.Module):
    def __init__(self, lfd_params, filename, backbone_id,
                 backbone_train=False, bottleneck_train=False, use_bottleneck=False):
        super().__init__()
        self.lfd_params = lfd_params
        self.backbone_id = backbone_id

        # parts of model to train
        self.backbone_train = backbone_train
        self.bottleneck_train = bottleneck_train

        self.use_bottleneck = use_bottleneck  # use to get features for IAD

        # model filenames
        self.filename = filename
        self.backbone_filename = ".".join([self.filename, "backbone", "pt"])
        self.bottleneck_filename = ".".join([self.filename, "bottleneck", "pt"])

        # model sections
        assert self.backbone_id in ["tsm", "i3d", "r21d", "eco", "pan", "vgg", "wrn", "trn"], \
            "ERROR: feature_extractor.py: backbone_id (" + self.backbone_id + ") not valid"

        pretrain_model_name = ""
        input_size = 0
        spatial_size = 7
        end_point = None

        # TSM
        if self.backbone_id == "tsm":
            from .backbone_model.backbone_tsm import BackboneTSM as Backbone
            pretrain_model_name = os.path.join(self.lfd_params.args.home_dir,
                                               "models/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth")
            input_size = 2048

        # TRN
        if self.backbone_id == "trn":
            from .backbone_model.backbone_trn import BackboneTRN as Backbone
            pretrain_model_name = os.path.join(self.lfd_params.args.home_dir,
                                               "models/TRN_somethingv2_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar")
            input_size = 2048

        # I3D
        elif self.backbone_id == "i3d":
            from .backbone_model.backbone_i3d import BackboneI3D as Backbone
            pretrain_model_name = os.path.join(self.lfd_params.args.home_dir,
                                               "models/rgb_imagenet.pt")
            input_size = 1024
            end_point = 5

        # R(2+1)D
        elif self.backbone_id == "r21d":
            from .backbone_model.backbone_r21d import BackboneR21D as Backbone
            input_size = 512
            spatial_size = 14

        # VGG
        elif self.backbone_id == "vgg":
            from .backbone_model.backbone_vgg import BackboneVGG as Backbone
            input_size = 512

        # Wide ResNet
        elif self.backbone_id == "wrn":
            from .backbone_model.backbone_wrn import BackboneWideResNet as Backbone
            input_size = 2048

        self.num_output_features = input_size
        self.backbone = Backbone(self.lfd_params, is_training=self.backbone_train, trim_model=use_bottleneck,
                                 filename=pretrain_model_name if self.backbone_train else self.backbone_filename,
                                 end_point=end_point)

        if self.use_bottleneck:
            self.bottleneck = SpatialBottleneck(self.lfd_params, is_training=self.bottleneck_train,
                                                filename=self.bottleneck_filename,
                                                bottleneck_size=self.lfd_params.args.bottleneck_size,
                                                input_size=input_size, spatial_size=spatial_size)
            self.num_output_features = self.lfd_params.args.bottleneck_size

    # Defining the forward pass
    def forward(self, x):
        x = self.backbone(x)
        if self.use_bottleneck:
            x = self.bottleneck(x)
        return x

    def save_model(self):
        if self.backbone_train:
            self.backbone.save_model(self.backbone_filename)
        if self.use_bottleneck and self.bottleneck_train:
            self.bottleneck.save_model(self.bottleneck_filename)
