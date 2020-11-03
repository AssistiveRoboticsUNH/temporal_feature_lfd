
def define_model(model_p):
    assert model_p in ["tsm", "i3d", "r21d"], "ERROR: exec_policy_learning_ditrl.py: model_p not defined"

    Classifier = None
    PolicyLearner = None
    num_segments = None
    bottleneck_size = None
    dense_sample = None
    dense_rate = None

    if model_p == "tsm":
        from model.classifier_ditrl_tsm import ClassifierDITRLTSM as Classifier
        from model.policylearner_ditrl_tsm import PolicyLearnerDITRLTSM as PolicyLearner
        num_segments = 16
        bottleneck_size = 16
        dense_sample = False
        dense_rate = 0
    elif model_p == "i3d":
        from model.classifier_ditrl_i3d import ClassifierDITRLI3D as Classifier
        from model.policylearner_ditrl_i3d import PolicyLearnerDITRLI3D as PolicyLearner
        num_segments = 64
        bottleneck_size = 8
        dense_sample = True
        dense_rate = 6
    elif model_p == "r21d":
        from model.classifier_ditrl_r21d import ClassifierDITRLR21D as Classifier
        from model.policylearner_ditrl_r21d import PolicyLearnerDITRLR21D as PolicyLearner
        num_segments = 64
        bottleneck_size = 8
        dense_sample = True
        dense_rate = 6

    return {"classifier": Classifier,
            "policy_learner": PolicyLearner,
            "num_segments": num_segments,
            "bottleneck_size": bottleneck_size,
            "dense_sample": dense_sample,
            "dense_rate": dense_rate
            }
