import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .spatial.spatial_ext_linear import SpatialExtLinear
from .temporal.temporal_pipeline import TemporalPipeline

from model_def import define_model


class ClassifierDITRL(nn.Module):
    def __init__(self, lfd_params, filename, backbone_id,
                 use_feature_extractor=False,
                 spatial_train=False, use_spatial=True,
                 ditrl_pipeline_train=False, use_pipeline=False,
                 return_iad=False, return_vee=False,
                 temporal_train=False, use_temporal=False, use_gcn=False, use_itr_lstm=False):
        super().__init__()

        self.lfd_params = lfd_params
        self.backbone_id = backbone_id

        # parts of model to train
        self.spatial_train = spatial_train
        self.ditrl_pipeline_train = ditrl_pipeline_train
        self.temporal_train = temporal_train

        self.use_feature_extractor = use_feature_extractor  # use to get features for IAD
        self.use_spatial = use_spatial  # use to get classification from IAD
        self.use_pipeline = use_pipeline  # use to get ITRs from IAD
        if self.use_pipeline:
            self.return_iad = return_iad
            self.return_vee = return_vee
        self.use_temporal = use_temporal  # use to learn from ITRs
        self.use_gcn = use_gcn
        self.use_itr_lstm = use_itr_lstm

        # model filenames
        self.filename = filename
        self.spatial_filename = ".".join([self.filename, "spatial", "pt"])
        self.pipeline_filename = ".".join([self.filename, "pipeline", "pk"])
        self.temporal_filename = ".".join([self.filename, "temporal", "pt"])

        # model sections
        if self.use_feature_extractor:
            self.feature_extractor = FeatureExtractor(lfd_params, filename, backbone_id,
                                                      backbone_train=spatial_train,
                                                      bottleneck_train=spatial_train,
                                                      use_bottleneck=True)

            if self.use_spatial:
                self.spatial = SpatialExtLinear(lfd_params, is_training=self.spatial_train,
                                                filename=self.spatial_filename,
                                                input_size=define_model(backbone_id)["bottleneck_size"],
                                                consensus="max", reshape_output=True)
        if self.use_pipeline:
            self.pipeline = TemporalPipeline(lfd_params, is_training=self.ditrl_pipeline_train,
                                             filename=self.pipeline_filename,
                                             return_iad=self.return_iad, return_vee=self.return_vee,
                                             use_gcn=self.use_gcn)
        if self.use_temporal:
            if self.use_gcn:
                from .temporal.temporal_ext_gcn import TemporalExtGCN
                self.temporal = TemporalExtGCN(lfd_params, is_training=self.temporal_train,
                                               filename=self.temporal_filename,
                                               node_size=lfd_params.args.bottleneck_size,
                                               num_relations=7,
                                               output_size=8)
            elif self.use_itr_lstm:
                from .temporal.temporal_ext_lstm import TemporalExtLSTM
                self.temporal = TemporalExtLSTM(lfd_params, is_training=self.temporal_train,
                                                  filename=self.filename,
                                                  input_size=lfd_params.args.bottleneck_size,
                                                  output_size=8)
            else:
                from .temporal.temporal_ext_linear import TemporalExtLinear
                self.temporal = TemporalExtLinear(lfd_params, is_training=self.temporal_train,
                                                  filename=self.temporal_filename,
                                                  input_size=(lfd_params.args.bottleneck_size ** 2 * 7),
                                                  output_size=8)

    # Defining the forward pass
    def forward(self, x):
        #print("x.shape0:", x.shape, self.use_temporal)
        if self.use_feature_extractor:
            x = self.feature_extractor(x)
            #print("x.shape0.1:", x.shape)
        if self.use_spatial:
            x = self.spatial(x)
            #print("x.shape0.2:", x.shape)
        if self.use_pipeline:
            x = self.pipeline(x)
            #print("x.shape0.3:", x.shape)
            #print("x0.3:", x)
        if self.use_temporal:
            x = self.temporal(x)
            #print("x.shape0.4:", x.shape)
            #print("x0.4:", x)

        #print("x:", x)
        #print("x.shape1:", x.shape)
        return x

    def save_model(self):
        if self.use_feature_extractor and self.spatial_train:
            self.feature_extractor.save_model()
            if self.use_spatial:
                self.spatial.save_model(self.spatial_filename)
        if self.use_pipeline and self.ditrl_pipeline_train:
            self.pipeline.save_model(self.pipeline_filename)
        if self.use_temporal and self.temporal_train:
            self.temporal.save_model(self.temporal_filename)

