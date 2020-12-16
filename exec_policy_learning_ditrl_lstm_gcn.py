import os
import pandas as pd
from parameter_parser import parse_model_args, default_model_args
from run_ditrl_pipeline import train_pipeline, generate_itr_files, generate_binarized_iad_files, generate_itr_files_gcn
from run_policy_learning import train, evaluate_single_action, evaluate_action_trace

from model.classifier_ditrl import ClassifierDITRL
from model.policy_learner_ditrl import PolicyLearnerDITRL

GENERATE_ITR=False
GENERATE_VEE=False
TRAIN = True
EVAL = True
FULL = True  # train backbone + DITRL at same time
#MODEL = "tsm"


def main(save_id, gen_itr, gen_vee, train_p, eval_p, backbone_id, full_p=False):
    print("save_id: {0}, train_p : {1}, eval_p: {2}, backbone_id: {3}, full_p: {4}".format(save_id, train_p, eval_p, backbone_id, full_p))

    if full_p:
        from exec_classifier_bottleneck import main as bottleneck_main
        bottleneck_main(save_id, train_p, eval_p, backbone_id)

    from model_def import define_model
    model_dict = define_model(backbone_id)

    num_segments = model_dict["num_segments"]
    bottleneck_size = model_dict["bottleneck_size"]
    dense_sample = model_dict["dense_sample"]
    dense_rate = model_dict["dense_rate"]

    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name,
                                    num_segments=num_segments, bottleneck_size=bottleneck_size,
                                    dense_sample=dense_sample, dense_rate=dense_rate)

    if gen_itr:

        print("Training Pipeline")
        model = ClassifierDITRL(lfd_params, filename, backbone_id, use_feature_extractor=True, use_spatial=False,
                                use_pipeline=True, use_temporal=False, spatial_train=False, ditrl_pipeline_train=True)
        model = train_pipeline(lfd_params, model)
        model.save_model()

        #print("model.pipeline.is_training:", model.pipeline.is_training)

        print("Generating ITR Files")
        generate_itr_files_gcn(lfd_params, model, "train", backbone=backbone_id)
        generate_itr_files_gcn(lfd_params, model, "evaluation", backbone=backbone_id)

    if gen_vee:
        model = ClassifierDITRL(lfd_params, filename, backbone_id, use_feature_extractor=True, use_spatial=False,
                                use_pipeline=True, use_temporal=False, spatial_train=False, ditrl_pipeline_train=False,
                                return_vee=True)

        print("Generating Sparse IAD Files")
        generate_binarized_iad_files(lfd_params, model, "train", backbone=backbone_id)
        generate_binarized_iad_files(lfd_params, model, "evaluation", backbone=backbone_id)

    if train_p:
        print("Training Policy")

        model = PolicyLearnerDITRL(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial=False,
                                   use_pipeline=False, use_temporal=True, spatial_train=False,
                                   ditrl_pipeline_train=False, temporal_train=True, policy_train=True, use_itr_lstm=True)
        model = train(lfd_params, model, input_dtype="gcn", verbose=True)  # make sure to use ITRs

        print("--------------")
        print("Saved Model")
        model.save_model()

    if eval_p:

        model = PolicyLearnerDITRL(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial=False,
                                   use_pipeline=False, use_temporal=True, spatial_train=False,
                                   ditrl_pipeline_train=False, temporal_train=False, use_itr_lstm=True)

        '''
        df = evaluate_single_action(lfd_params, model, input_dtype="itr")
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_single_action.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)
        '''

        df = evaluate_action_trace(lfd_params, model, input_dtype="gcn")
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_action_trace.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

        df = evaluate_action_trace(lfd_params, model, input_dtype="gcn", ablation=True, verbose=True, mode="train")
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_action_trace_ablation_train.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

        df = evaluate_action_trace(lfd_params, model, input_dtype="itr", ablation=True, verbose=True, mode="evaluation")
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_action_trace_ablation_eval.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)


if __name__ == '__main__':

    import sys
    model_p = sys.argv[1]
    FULL = int(sys.argv[2])

    #save_id = "policy_learning_ditrl_"+MODEL  # "policy_learning_ditrl_tsm_bn16_2"
    if model_p == "tsm":
        save_id = "classifier_bottleneck_tsm3"
    elif model_p == "vgg":
        save_id = "classifier_bottleneck_vgg0"
    elif model_p == "wrn":
        save_id = "classifier_bottleneck_wrn1"
    elif model_p == "r21d":
        save_id = "classifier_bottleneck_r21d0"
    elif model_p == "i3d":
        save_id = "classifier_bottleneck_i3d0"

    new_save_id = "policy_learning_ditrl_"+model_p
    old_save_dir = os.path.join("base_models", save_id)
    new_save_dir = os.path.join("saved_models", new_save_id)
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)

        from shutil import copy2
        for f in os.listdir(old_save_dir):
            copy2(os.path.join(old_save_dir, f), new_save_dir)
    save_id = new_save_id

    main(save_id, GENERATE_ITR, GENERATE_VEE, TRAIN, EVAL, model_p, full_p=FULL)
