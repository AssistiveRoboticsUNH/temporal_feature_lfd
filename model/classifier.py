import torch
import os
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .spatial.spatial_ext_linear import SpatialExtLinear
from .spatial.spatial_ext_lstm import SpatialExtLSTM
from .spatial.spatial_ext_tcn import SpatialExtTCN
from .temporal.temporal_pipeline import TemporalPipeline
from .temporal.temporal_ext_gcn import TemporalExtGCN

from enums import *


class Classifier(nn.Module):
    def __init__(self, lfd_params, filename, backbone_id, suffix,

                 use_feature_extractor=False, train_feature_extractor=False,
                 use_bottleneck=False,
                 use_spatial=False, train_spatial=False,
                 use_pipeline=False, train_pipeline=False,
                 use_temporal=False, train_temporal=False,

                 policy_learn_ext=False
                 ):

        super().__init__()

        self.lfd_params = lfd_params
        self.backbone_id = backbone_id

        # model parts to use
        self.use_feature_extractor = use_feature_extractor
        self.use_bottleneck = use_bottleneck
        self.use_spatial = use_spatial
        self.use_pipeline = use_pipeline
        self.use_temporal = use_temporal

        # parts of model to train
        self.train_feature_extractor = train_feature_extractor
        self.train_spatial = train_spatial
        self.train_pipeline = train_pipeline
        self.train_temporal = train_temporal

        # use policy learning architecture
        self.policy_learn_ext = policy_learn_ext

        # get the number of frames expected from the IAD
        self.num_frames = self.lfd_params.model.iad_frames

        # if using bottleneck modify number of features
        self.num_features = self.lfd_params.model.original_size
        if suffix not in [Suffix.GENERATE_IAD, Suffix.LINEAR, Suffix.LSTM]:
            self.use_bottleneck = True
            self.num_features = self.lfd_params.model.bottleneck_size

        # model filenames
        self.filename = os.path.join(self.lfd_params.model_save_dir, filename)

        # Model Layers
        if self.use_feature_extractor:
            self.feature_extractor = FeatureExtractor(lfd_params, self.filename, backbone_id,
                                                      backbone_train=self.train_feature_extractor,
                                                      bottleneck_train=self.train_feature_extractor,
                                                      use_bottleneck=self.use_bottleneck)

        # parameter is updated with information from the application
        output_size = len(self.lfd_params.application.obs_label_list)

        # select model modifications
        if suffix in [Suffix.BACKBONE, Suffix.LINEAR, Suffix.LINEAR_IAD]:
            input_size = self.num_features if suffix == Suffix.BACKBONE else self.num_features * self.num_frames
            consensus = "max" if suffix == Suffix.BACKBONE else "flat"

            self.spatial = SpatialExtLinear(lfd_params, is_training=self.train_spatial,
                                            filename=self.filename,
                                            input_size=input_size,
                                            output_size=output_size,
                                            consensus=consensus,
                                            reshape_output=True)

        elif suffix in [Suffix.LSTM, Suffix.LSTM_IAD]:
            self.spatial = SpatialExtLSTM(lfd_params, is_training=self.train_spatial,
                                          filename=self.filename,
                                          input_size=self.num_features,
                                          output_size=output_size,
                                          consensus=None)

        elif suffix in [Suffix.TCN]:
            self.spatial = SpatialExtTCN(lfd_params, is_training=self.train_spatial,
                                         filename=self.filename,
                                         input_size=self.num_features,
                                         output_size=output_size,
                                         consensus=None)

        elif suffix == Suffix.PIPELINE:
            self.pipeline = TemporalPipeline(lfd_params, is_training=self.train_pipeline,
                                             filename=self.filename,
                                             use_gcn=True)

        elif suffix == Suffix.DITRL:
            self.temporal = TemporalExtGCN(lfd_params, is_training=self.train_temporal,
                                           filename=self.filename,
                                           node_size=lfd_params.model.bottleneck_size,
                                           num_relations=1,
                                           output_size=output_size)

    # Defining the forward pass
    def forward(self, x):

        # parameter is used to reshape the data to account fro a batch of several videos
        if self.use_spatial:
            history_length = x.shape[0]
            if self.policy_learn_ext:
                history_length = x.shape[1]

        # pass through only the necessary layers
        if self.use_feature_extractor:
            x = self.feature_extractor(x)

        if self.use_spatial:
            x = x.view(history_length, -1, self.num_features)
            x = self.spatial(x)

        if self.use_pipeline:
            x = self.pipeline(x)

        if self.use_temporal:
            x = self.temporal(x)

        return x

    # Save all parts of the model
    def save_model(self):
        if self.use_feature_extractor and self.train_feature_extractor:
            self.feature_extractor.save_model()

        if self.use_spatial and self.train_spatial:
            self.spatial.save_model()

        if self.use_pipeline and self.train_pipeline:
            self.pipeline.save_model()

        if self.use_temporal and self.train_temporal:
            self.temporal.save_model()
