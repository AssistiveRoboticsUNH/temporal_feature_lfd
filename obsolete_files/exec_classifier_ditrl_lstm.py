import os
import pandas as pd
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from run_ditrl_pipeline import train_pipeline, generate_itr_files

from model.classifier_ditrl import ClassifierDITRL

GENERATE_ITR = False
TRAIN = True
EVAL = True
FULL = False  # train backbone + DITRL at same time
MODEL = "tsm"


def main(save_id, gen_p, train_p, eval_p, backbone_id, full_p=False):

    if full_p:
        from exec_classifier_bottleneck import main as backbone_main
        backbone_main(save_id, train_p, eval_p, backbone_id)

    from model_def import define_model
    model_dict = define_model(backbone_id)

    num_segments = model_dict["num_segments"]
    bottleneck_size = model_dict["bottleneck_size"]
    dense_sample = model_dict["dense_sample"]
    dense_rate = model_dict["dense_rate"]

    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "../model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name, num_segments=num_segments, bottleneck_size=bottleneck_size,
                                    dense_sample=dense_sample, dense_rate=dense_rate)

    if gen_p:

        print("Training Pipeline")
        model = ClassifierDITRL(lfd_params, filename, backbone_id, use_feature_extractor=True, use_spatial=False,
                                use_pipeline=True, use_temporal=False, spatial_train=False, ditrl_pipeline_train=True)
        model = train_pipeline(lfd_params, model)
        model.save_model()

        print("Generating ITR Files")
        generate_itr_files(lfd_params, model, "train")
        generate_itr_files(lfd_params, model, "evaluation")

    if train_p:
        model = ClassifierDITRL(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial=False,
                                use_pipeline=False, use_temporal=True, spatial_train=False, ditrl_pipeline_train=False,
                                temporal_train=True, use_itr_lstm=True)
        model = train(lfd_params, model, input_dtype="itr", verbose=True)  # make sure to use ITRs
        model.save_model()

    if eval_p:
        print("Evaluating Model")
        model = ClassifierDITRL(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial=False,
                                use_pipeline=False, use_temporal=True, spatial_train=False, ditrl_pipeline_train=False,
                                temporal_train=False, use_itr_lstm=True)

        train_df = evaluate(lfd_params, model, mode="train", input_dtype="itr")
        train_df["mode"] = ["train"] * len(train_df)
        eval_df = evaluate(lfd_params, model, mode="evaluation", verbose=True, input_dtype="itr")
        eval_df["mode"] = ["evaluation"] * len(eval_df)
        df = pd.concat([train_df, eval_df])
        df["repeat"] = [save_id] * len(df)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + ".csv")
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

    new_save_id = "classifier_ditrl_lstm_"+model_p
    old_save_dir = os.path.join("base_models", save_id)
    new_save_dir = os.path.join("saved_models", new_save_id)
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)

        from shutil import copy2
        for f in os.listdir(old_save_dir):
            copy2(os.path.join(old_save_dir, f), new_save_dir)
    save_id = new_save_id

    main(save_id, GENERATE_ITR, TRAIN, EVAL, model_p, full_p=FULL)
