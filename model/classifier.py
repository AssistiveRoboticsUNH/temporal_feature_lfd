import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .spatial.spatial_ext_linear import SpatialExtLinear
from .spatial.spatial_ext_lstm import SpatialExtLSTM

from model_def import define_model


class Classifier(nn.Module):
    def __init__(self, lfd_params, filename, backbone_id,
                 feature_extractor_train=False, use_feature_extractor=False, spatial_train=False, use_spatial=False,
                 use_spatial_lstm=False):
        super().__init__()

        self.lfd_params = lfd_params
        self.backbone_id = backbone_id

        # parts of model to train
        self.feature_extractor_train = feature_extractor_train
        self.spatial_train = spatial_train

        self.use_feature_extractor = use_feature_extractor  # use to get features for IAD
        self.use_spatial = use_spatial  # use to get classification from IAD
        self.use_spatial_lstm = use_spatial_lstm

        # model filenames
        self.filename = filename
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])

        #self.num_features = define_model(backbone_id)["original_size"]
        self.num_features = define_model(backbone_id)["bottleneck_size"]
        self.num_frames = define_model(backbone_id)["iad_frames"]

        # model sections
        if self.use_feature_extractor:
            self.feature_extractor = FeatureExtractor(lfd_params, filename, backbone_id,
                                                      backbone_train=self.feature_extractor_train,
                                                      bottleneck_train=self.feature_extractor_train,
                                                      use_bottleneck=True)
        if self.use_spatial:
            self.spatial = SpatialExtLinear(lfd_params, is_training=self.spatial_train,
                                            filename=self.spatial_filename,
                                            input_size=self.num_features * self.num_frames,  # self.num_features,
                                            consensus="flat")
        elif self.use_spatial_lstm:
            self.spatial = SpatialExtLSTM(lfd_params, is_training=self.spatial_train,
                                          filename=self.filename,
                                          input_size=self.num_features,
                                          consensus=None)

    # Defining the forward pass
    def forward(self, x):
        history_length = x.shape[1]
        if self.use_feature_extractor:
            x = self.feature_extractor(x)
        if self.use_spatial or self.use_spatial_lstm:
            x = x.view(history_length, -1, self.num_features)
            x = self.spatial(x)
            x = torch.squeeze(x, 1)
        return x

    def save_model(self):
        if self.use_feature_extractor and self.feature_extractor_train:
            self.feature_extractor.save_model()
        if (self.use_spatial or self.use_spatial_lstm) and self.spatial_train:
            self.spatial.save_model(self.spatial_filename)
