import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from run_classification import generate_iad_files
import pandas as pd

from model.classifier import Classifier

TRAIN = True
EVAL = True
MODEL = "tsm"


#def main(save_id, train_p, eval_p, backbone_id):
def main(save_id, gen_p, train_p, eval_p, backbone_id, return_eval=False, use_bottleneck=True, file_id=""):
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

    if gen_p:
        print("Generating ITR Files")
        model = Classifier(lfd_params, filename, backbone_id, use_feature_extractor=True, use_spatial=True,
                           spatial_train=False, use_bottleneck=use_bottleneck)

        generate_iad_files(lfd_params, model, "train", backbone=backbone_id)
        generate_iad_files(lfd_params, model, "evaluation", backbone=backbone_id)

    if train_p:
        model = Classifier(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial=True,
                           spatial_train=True, use_bottleneck=use_bottleneck)

        model = train(lfd_params, model, verbose=True)
        model.save_model()

    if eval_p:
        model = Classifier(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial=True,
                           spatial_train=False, use_bottleneck=use_bottleneck)

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