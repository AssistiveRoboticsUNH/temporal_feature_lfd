import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .spatial.spatial_ext_linear import SpatialExtLinear


class Classifier(nn.Module):
    def __init__(self, lfd_params, filename, backbone_id,
                 spatial_train=False):
        super().__init__()

        # parts of model to train
        self.spatial_train = spatial_train

        # model filenames
        self.filename = filename
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])

        # model sections
        self.feature_extractor = FeatureExtractor(lfd_params, filename, backbone_id, backbone_train=spatial_train)
        self.spatial = SpatialExtLinear(lfd_params, is_training=self.spatial_train,
                                        filename=self.spatial_filename,
                                        input_size=self.feature_extractor.num_output_features,
                                        consensus="max")

    # Defining the forward pass
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.spatial(x)
        return x

    def save_model(self):
        if self.spatial_train:
            self.feature_extractor.save_model()
            self.spatial.save_model(self.spatial_filename)
