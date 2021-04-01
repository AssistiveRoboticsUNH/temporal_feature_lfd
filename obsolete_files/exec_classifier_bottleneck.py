import os
from parameter_parser import default_model_args
from run_classification import train, evaluate
import pandas as pd

from obsolete_files.classifier_ditrl import ClassifierDITRL

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
    filename = os.path.join(dir_name, "../model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name,
                                    num_segments=num_segments, bottleneck_size=bottleneck_size,
                                    dense_sample=dense_sample, dense_rate=dense_rate)  # parse_model_args()

    if train_p:
        model = ClassifierDITRL(lfd_params, filename, backbone_id, use_feature_extractor=True, use_spatial=True,
                                spatial_train=True)

        model = train(lfd_params, model, verbose=True)
        model.save_model()

    if eval_p:
        model = ClassifierDITRL(lfd_params, filename, backbone_id, use_feature_extractor=True, use_spatial=True,
                                spatial_train=False)

        train_df = evaluate(lfd_params, model, mode="train")
        train_df["mode"] = ["train"]*len(train_df)
        eval_df = evaluate(lfd_params, model, mode="evaluation", verbose=True)
        eval_df["mode"] = ["evaluation"] * len(eval_df)
        df = pd.concat([train_df, eval_df])

        if return_eval:
            return df

        df["repeat"] = ["1"]*len(df)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_spatial.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)


if __name__ == '__main__':
    save_id = "classifier_bottleneck_"+MODEL
    main(save_id, TRAIN, EVAL, MODEL)