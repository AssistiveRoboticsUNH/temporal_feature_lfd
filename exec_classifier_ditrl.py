import os
import pandas as pd
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from run_ditrl_pipeline import train_pipeline, generate_itr_files
from model.classifier_ditrl_tsm import ClassifierDITRLTSM

TRAIN = True
EVAL = True
FULL = False  # train backbone + DITRL at same time
MODEL = "tsm"


def main(save_id, train_p, eval_p, model_p, full_p=False):
    assert model_p in ["tsm", "i3d"], "ERROR: exec_policy_learning_backbone.py: model_p not defined"

    if full_p:
        from exec_classifier_bottleneck import main as backbone_main
        backbone_main(save_id, train_p, eval_p, model_p)

    if model_p == "tsm":
        from model.classifier_ditrl_tsm import ClassifierDITRLTSM as Classifier
        num_segments = 16
        bottleneck_size = 16
        dense_sample = False
    elif model_p == "i3d":
        from model.classifier_ditrl_i3d import ClassifierDITRLI3D as Classifier
        num_segments = 64
        bottleneck_size = 8
        dense_sample = True
        dense_rate = 6

    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name, num_segments=num_segments, bottleneck_size=bottleneck_size,
                                    dense_sample=dense_sample, dense_rate=dense_rate)

    if train_p:

        print("Training Pipeline")
        model = Classifier(lfd_params, filename, use_feature_extractor=True, use_spatial=False, use_pipeline=True, use_temporal=False,
                                   spatial_train=False, ditrl_pipeline_train=True)
        model = train_pipeline(lfd_params, model)
        model.save_model()

        print("Generating ITR Files")
        generate_itr_files(lfd_params, model, "train")
        generate_itr_files(lfd_params, model, "evaluation")

        model = Classifier(lfd_params, filename, use_feature_extractor=False, use_spatial=False,
                                   use_pipeline=False, use_temporal=True,
                                   spatial_train=False, ditrl_pipeline_train=False, temporal_train=True)
        model = train(lfd_params, model, input_dtype="itr", verbose=True)  # make sure to use ITRs
        model.save_model()

    if eval_p:
        print("Evaluating Model")
        model = Classifier(lfd_params, filename, use_feature_extractor=False, use_spatial=False,
                                   use_pipeline=False, use_temporal=True,
                                   spatial_train=False, ditrl_pipeline_train=False, temporal_train=False)

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

    save_id_p = "classifier_ditrl_"+MODEL
    main(save_id_p, TRAIN, EVAL, MODEL, full_p=FULL)
