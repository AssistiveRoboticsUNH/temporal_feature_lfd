import os
import pandas as pd
from parameter_parser import parse_model_args, default_model_args
from run_ditrl_pipeline import train_pipeline, generate_itr_files
from run_policy_learning import train, evaluate_single_action, evaluate_action_trace

TRAIN = True
EVAL = True
FULL = False  # train backbone + DITRL at same time
MODEL = "tsm"


def main(save_id, train_p, eval_p, model_p, full_p=False):
    assert model_p in ["tsm", "i3d"], "ERROR: exec_policy_learning_ditrl.py: model_p not defined"

    if full_p:
        from exec_classifier_backbone import main as backbone_main
        backbone_main(save_id, train_p, eval_p, model_p)

    if model_p == "tsm":
        from model.classifier_ditrl_tsm import ClassifierDITRLTSM as Classifier
        from model.policylearner_ditrl_tsm import PolicyLearnerDITRLTSM as PolicyLearner
        num_segments = 16
        bottleneck_size = 16
        dense_sample = False
    elif model_p == "i3d":
        from model.classifier_ditrl_i3d import ClassifierDITRLI3D as Classifier
        from model.policylearner_ditrl_i3d import PolicyLearnerDITRLI3D as PolicyLearner
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
                                    dense_sample=dense_sample, dense_rate=dense_rate)

    if train_p:
        print("Training Pipeline")
        model = Classifier(lfd_params, filename, use_feature_extractor=True, use_spatial=False, use_pipeline=True, use_temporal=False,
                                   spatial_train=False, ditrl_pipeline_train=True)
        model = train_pipeline(lfd_params, model)
        model.save_model()

        #print("model.pipeline.is_training:", model.pipeline.is_training)

        print("Generating ITR Files")
        generate_itr_files(lfd_params, model, "train", backbone="tsm")
        generate_itr_files(lfd_params, model, "evaluation", backbone="tsm")

        print("Training Policy")

        model = PolicyLearner(lfd_params, filename, use_feature_extractor=False, use_spatial=False, use_pipeline=False, use_temporal=True,
                                   spatial_train=False, ditrl_pipeline_train=False, temporal_train=True)
        model = train(lfd_params, model, input_dtype="itr", verbose=True)  # make sure to use ITRs
        model.save_model()

    if eval_p:

        model = PolicyLearner(lfd_params, filename, use_feature_extractor=False, use_spatial=False,
                                      use_pipeline=False, use_temporal=True,
                                      spatial_train=False, ditrl_pipeline_train=False, temporal_train=False)
        '''
        df = evaluate_single_action(lfd_params, model, input_dtype="itr")

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_single_action.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

        df = evaluate_action_trace(lfd_params, model, input_dtype="itr")

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_action_trace.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)
        '''
        df = evaluate_action_trace(lfd_params, model, input_dtype="itr", ablation=True, verbose=True, mode="train")
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_action_trace_ablation_train.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

        df = evaluate_action_trace(lfd_params, model, input_dtype="itr", ablation=True, verbose=True, mode="evaluation")
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_action_trace_ablation_eval.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)



if __name__ == '__main__':

    save_id = "policy_learning_ditrl_"+MODEL  # "policy_learning_ditrl_tsm_bn16_2"
    main(save_id, TRAIN, EVAL, MODEL, full_p=FULL)