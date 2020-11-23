import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from run_classification import generate_iad_files
import pandas as pd

from model.classifier import Classifier

TRAIN = True
EVAL = True
MODEL = "tsm"


def main(save_id, train_p, eval_p, backbone_id, return_eval=False):
    print("save_id: {0}, train_p : {1}, eval_p: {2}, backbone_id: {3}, ".format(save_id, train_p, eval_p, backbone_id))

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
                                    dense_sample=dense_sample, dense_rate=dense_rate)  # parse_model_args()

    if train_p:
        print("Generating ITR Files")
        model = Classifier(lfd_params, filename, backbone_id, use_feature_extractor=True, use_spatial_lstm=False,
                           spatial_train=False)

        generate_iad_files(lfd_params, model, "train", backbone=backbone_id)
        generate_iad_files(lfd_params, model, "evaluation", backbone=backbone_id)

        model = Classifier(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial_lstm=True,
                                spatial_train=True)

        model = train(lfd_params, model, verbose=True, input_dtype="iad")
        model.save_model()

    if eval_p:
        model = Classifier(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial_lstm=True,
                                spatial_train=False)

        train_df = evaluate(lfd_params, model, mode="train", input_dtype="iad")
        train_df["mode"] = ["train"]*len(train_df)
        eval_df = evaluate(lfd_params, model, mode="evaluation", verbose=True, input_dtype="iad")
        eval_df["mode"] = ["evaluation"] * len(eval_df)
        df = pd.concat([train_df, eval_df])

        if return_eval:
            return df

        df["repeat"] = ["1"]*len(df)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_spatial.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)


if __name__ == '__main__':

    import sys
    model_p = sys.argv[1]

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

    new_save_id = "classifier_lstm_" + model_p
    old_save_dir = os.path.join("base_models", save_id)
    new_save_dir = os.path.join("saved_models", new_save_id)
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)

        from shutil import copy2

        for f in os.listdir(old_save_dir):
            copy2(os.path.join(old_save_dir, f), new_save_dir)
    save_id = new_save_id

    main(save_id, TRAIN, EVAL, model_p)