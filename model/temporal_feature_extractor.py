import os
import pickle
import numpy as np
import torch

from .feature_extractor import FeatureExtractor
from .ditrl import DITRL_Pipeline, DITRL_Linear


class TemporalFeatureExtractor(FeatureExtractor):

    """
    use_pipeline - flag that if set to True will take frames as input and pass teh result through the D-ITR-L pipeline.
    If false then the model will not pass data through the D-ITR-L and expects an ITR input.

    use_model - flag that if set to True expects ITRs as input and passes them through a D-ITR-L model
    """

    def __init__(self, lfd_params,
                 use_pipeline=True, train_pipeline=False,
                 use_model=True, train_model=False):
        super().__init__(lfd_params, train_pipeline)

        self.use_pipeline = use_pipeline
        self.use_model = use_model

        assert self.use_pipeline or self.use_model, \
            "temporal_feature_extractor.py: D-ITR-L should be run with the 'use_pipeline' AND/OR the 'use_model' flags"

        num_features = self.bottleneck_size
        num_classes = self.num_classes

        # Setup the D-ITR-L pipeline
        if self.use_pipeline:

            self.pipeline_filename = self.lfd_params.generate_ditrl_modelname()
            if not train_pipeline:
                assert os.path.exists(self.pipeline_filename), \
                    "temporal_feature_extractor.py: Cannot find D-ITR-L Pipeline Saved Model"
                self.pipeline = pickle.load(self.pipeline_filename)

            else:
                self.pipeline = DITRL_Pipeline(num_features)

            self.pipeline.is_training = train_pipeline

        # Setup the D-ITR-L model
        if self.use_model:
            model_filename = self.lfd_params.generate_ditrl_ext_modelname()
            self.model = DITRL_Linear(num_features, num_classes, train_model, model_filename)

    def forward(self, inp, cleanup=True):

        if self.use_pipeline:
            # pass data through CNNs
            # ---
            activation_map = self.rgb_net(inp)

            # apply linear layer and consensus module to the output of the CNN
            activation_map = activation_map.view((-1, self.rgb_net.num_segments) + activation_map.size()[1:])

            # detach activation map from pyTorch and convert to NumPy array
            activation_map = activation_map.detach().cpu().numpy()

            # pass data through D-ITR-L Pipeline
            # ---
            itr_out = []
            batch_num = activation_map.shape[0]
            for i in range(batch_num):
                itr = self.pipeline.convert_activation_map_to_itr(activation_map[i], cleanup=cleanup)
                itr_out.append(itr)
            itr_out = np.array(itr_out)

            # pre-process ITRS
            # scale / TFIDF

            # evaluate on ITR
            inp = torch.autograd.Variable(torch.from_numpy(itr_out).cuda())

        if self.use_model:
            # pass ITRs through D-ITR-L Model
            # ---
            inp = self.model(inp)

        return inp

    def save_model(self):
        super().save_model()

        if self.use_pipeline:
            self.fit_pipeline()
            with open(self.pipeline_filename, "wb") as f:
                pickle.dump(self.pipeline, f)

        if self.use_model:
            self.model.save_model()

    def fit_pipeline(self):
        self.pipeline.fit_tfidf()

