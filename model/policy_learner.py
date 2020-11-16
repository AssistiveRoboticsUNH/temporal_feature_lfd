import torch

from .classifier import Classifier
from .policy.policy_lstm import PolicyLSTM


class PolicyLearner(Classifier):
    def __init__(self, lfd_params, filename, backbone_id,
                 backbone_train=False, bottleneck_train=False, use_bottleneck=False, policy_train=False):
        super().__init__(lfd_params, filename, backbone_id,
                         backbone_train=backbone_train, bottleneck_train=bottleneck_train, use_bottleneck=use_bottleneck)

        # parts of model to train
        self.policy_train = policy_train

        # model filenames
        self.lstm_filename = ".".join([self.filename, "lstm", "pt"])
        self.fc_filename = ".".join([self.filename, "policy", "pt"])

        # model sections
        self.policy = PolicyLSTM(lfd_params, is_training=policy_train,
                                 lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)

    # Defining the forward pass
    def forward(self, obs_x, act_x):
        obs_x = super().forward(obs_x)
        obs_x = torch.unsqueeze(obs_x, 0)

        x = self.policy(obs_x, act_x)
        return x

    def save_model(self):
        super().save_model()
        if self.policy_train:
            self.policy.save_model(lstm_filename=self.lstm_filename, fc_filename=self.fc_filename)
