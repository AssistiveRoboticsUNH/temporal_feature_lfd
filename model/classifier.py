import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .spatial.spatial_ext_linear import SpatialExtLinear
from .spatial.spatial_ext_lstm import SpatialExtLSTM
from .temporal.temporal_pipeline import TemporalPipeline
from .temporal.temporal_ext_gcn import TemporalExtGCN

from model_def import define_model
from enums import *


class Classifier(nn.Module):
    def __init__(self, lfd_params, filename, backbone_id, suffix,

                 use_feature_extractor=False, train_feature_extractor=False,
                 use_spatial=False, train_spatial=False,
                 use_pipeline=False, train_pipeline=False,
                 use_temporal=False, train_temporal=False,

                 use_bottleneck=True,
                 #use_spatial_lstm=False,
                 policy_learn_ext=False

                 ):

        super().__init__()

        self.lfd_params = lfd_params
        self.backbone_id = backbone_id

        # parts of model to train
        self.train_feature_extractor = train_feature_extractor
        self.train_spatial = train_spatial

        self.use_feature_extractor = use_feature_extractor  # use to get features for IAD
        self.use_spatial = use_spatial  # use to get classification from IAD
        self.use_spatial_lstm = use_spatial_lstm
        self.policy_learn_ext = policy_learn_ext

        self.use_bottleneck = False
        if suffix in [Suffix.LINEAR_IAD, Suffix.LSTM_IAD, Suffix.DITRL, Suffix.PIPELINE]:
            self.use_bottleneck = True

        # model filenames
        self.filename = filename
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])
        self.temporal_filename = ".".join([self.filename, "temporal", "pt"])

        if self.use_bottleneck:
            self.num_features = define_model(backbone_id)["bottleneck_size"]
        else:
            self.num_features = define_model(backbone_id)["original_size"]
        self.num_frames = define_model(backbone_id)["iad_frames"]

        # Model Layers
        if self.use_feature_extractor:
            self.feature_extractor = FeatureExtractor(lfd_params, filename, backbone_id,
                                                      backbone_train=self.train_feature_extractor,
                                                      bottleneck_train=self.train_feature_extractor,
                                                      use_bottleneck=self.use_bottleneck)

        output_size = 8  # update with information from the application
        if suffix == Suffix.LINEAR or suffix == Suffix.LINEAR_IAD:

            self.spatial = SpatialExtLinear(lfd_params, is_training=self.train_spatial,
                                            filename=self.spatial_filename,
                                            input_size=self.num_features * self.num_frames,
                                            output_size=output_size,
                                            consensus="flat", reshape_output=True)

        elif suffix == Suffix.LSTM or suffix == Suffix.LSTM_IAD:
            self.spatial = SpatialExtLSTM(lfd_params, is_training=self.train_spatial,
                                          filename=self.filename,
                                          input_size=self.num_features,
                                          output_size=output_size,
                                          consensus=None)

        elif suffix == Suffix.DITRL:
            if self.use_pipeline:
                assert self.use_spatial, "classifier.py: use_spatial parameter must be set to 'True' when use_pipeline"

                self.spatial = SpatialExtLinear(lfd_params, is_training=False,
                                                filename=self.spatial_filename,
                                                input_size=self.num_features,
                                                consensus="max", reshape_output=True)
                self.pipeline = TemporalPipeline(lfd_params, is_training=self.train_pipeline,
                                                 filename=self.pipeline_filename,
                                                 # return_iad=self.return_iad, return_vee=self.return_vee,
                                                 use_gcn=True)

            self.temporal = TemporalExtGCN(lfd_params, is_training=self.train_temporal,
                                           filename=self.temporal_filename,
                                           node_size=lfd_params.args.bottleneck_size,
                                           num_relations=7,
                                           output_size=output_size)

    # Defining the forward pass
    def forward(self, x):

        if self.policy_learn_ext:
            history_length = x.shape[1]
        else:
            history_length = x.shape[0]

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

        if (self.use_spatial or self.use_spatial_lstm) and self.train_spatial:
            self.spatial.save_model(self.spatial_filename)

        if self.use_pipeline and self.ditrl_train_pipeline:
            self.pipeline.save_model(self.pipeline_filename)
        if self.use_temporal and self.train_temporal:
            self.temporal.save_model(self.temporal_filename)
