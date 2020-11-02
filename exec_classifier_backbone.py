import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
import pandas as pd

TRAIN = True
EVAL = True
MODEL = "tsm"


def main(save_id, train_p, eval_p, model_p):
    assert model_p in ["tsm", "i3d"], "ERROR: exec_classifier_backbone.py: model_p not defined"

    if model_p == "tsm":
        from model.classifier_backbone_tsm import ClassifierBackboneTSM as Classifier
        num_segments = 16
        bottleneck_size = 16
        dense_sample = False
    elif model_p == "i3d":
        from model.classifier_backbone_i3d import ClassifierBackboneI3D as Classifier
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
        model = Classifier(lfd_params, filename, spatial_train=True)

        model = train(lfd_params, model, verbose=True)
        model.save_model()

    if eval_p:
        model = Classifier(lfd_params, filename, spatial_train=False)

        train_df = evaluate(lfd_params, model, mode="train")
        train_df["mode"] = ["train"]*len(train_df)
        eval_df = evaluate(lfd_params, model, mode="evaluation", verbose=True)
        eval_df["mode"] = ["evaluation"] * len(eval_df)
        df = pd.concat([train_df, eval_df])
        df["repeat"] = ["1"]*len(df)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_spatial.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

if __name__ == '__main__':
    save_id = "classifier_backbone_"+MODEL
    main(save_id, TRAIN, EVAL, MODEL)