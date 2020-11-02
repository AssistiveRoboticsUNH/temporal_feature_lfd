import os
from parameter_parser import parse_model_args, default_model_args
from run_policy_learning import train, evaluate_single_action, evaluate_action_trace

TRAIN = True
EVAL = True
MODEL = "tsm"


def main(save_id, train_p, eval_p, model_p):
    assert model_p in ["tsm", "i3d"], "ERROR: exec_policy_learning_backbone.py: model_p not defined"

    if model_p == "tsm":
        from model.classifier_backbone_tsm import ClassifierBackboneTSM as Classifier
        from model.policylearner_backbone_tsm import PolicyLearnerBackboneTSM as PolicyLearner
        num_segments = 16
        bottleneck_size = 16
        dense_sample = False
    elif model_p == "i3d":
        from model.classifier_backbone_i3d import ClassifierBackboneI3D as Classifier
        from model.policylearner_backbone_tsm import PolicyLearnerBackboneI3D as PolicyLearner
        num_segments = 64
        bottleneck_size = 8
        dense_sample = True
        dense_rate = 6

    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name,
                                    num_segments=num_segments, bottleneck_size=bottleneck_size,
                                    dense_sample=dense_sample, dense_rate=dense_rate)  # parse_model_args()

    if train_p:
        model = PolicyLearner(lfd_params, filename, spatial_train=True, policy_train=True)

        model = train(lfd_params, model, verbose=True)
        model.save_model()

    if eval_p:
        model = PolicyLearner(lfd_params, filename, spatial_train=False, policy_train=False)

        df = evaluate_single_action(lfd_params, model, verbose=True)
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_single_action.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

        df = evaluate_action_trace(lfd_params, model, verbose=True)
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_action_trace.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

        df = evaluate_action_trace(lfd_params, model, mode="train", ablation=True, verbose=True)
        out_filename = os.path.join(lfd_params.args.output_dir,
                                    "output_" + save_id + "_action_trace_ablation.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)


if __name__ == '__main__':
    save_id = "policy_learning_backbone_"+MODEL
    main(save_id, TRAIN, EVAL, MODEL)