import torch

from .classifier import Classifier
from .policy.policy_lstm import PolicyLSTM


class PolicyLearner(Classifier):
    def __init__(self, lfd_params, filename, backbone_id, suffix, 
                 use_feature_extractor=False, train_feature_extractor=False,
                 use_bottleneck=False,
                 use_spatial=False, train_spatial=False,
                 use_pipeline=False, train_pipeline=False,
                 use_temporal=False, train_temporal=False,
                 train_policy=False):
        super().__init__(lfd_params, filename, backbone_id, suffix, 
                         use_feature_extractor=use_feature_extractor, train_feature_extractor=train_feature_extractor,
                         use_bottleneck=use_bottleneck,
                         use_spatial=use_spatial, train_spatial=train_spatial,
                         use_pipeline=use_pipeline, train_pipeline=train_pipeline,
                         use_temporal=use_temporal, train_temporal=train_temporal,
                         policy_learn_ext=True)

        # parts of model to train
        self.train_policy = train_policy

        # model sections
        input_size = len(lfd_params.application.obs_label_list) + len(lfd_params.application.act_label_list)
        self.policy = PolicyLSTM(self.lfd_params, is_training=self.train_policy,
                                 input_size=input_size,
                                 filename=self.filename)

    # Defining the forward pass
    def forward(self, obs_x, act_x):
        obs_x = super().forward(obs_x)
        obs_x = torch.unsqueeze(obs_x, 0)

        x = self.policy(obs_x, act_x)
        return x

    def save_model(self):
        super().save_model()
        if self.train_policy:
            self.policy.save_model()
