import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from model.classifier_backbone_tsm import ClassifierBackboneTSM

TRAIN = False
EVAL = True

if __name__ == '__main__':

    save_id = "classifier_backbone_tsm"
    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name)  # parse_model_args()

    if TRAIN:
        model = ClassifierBackboneTSM(lfd_params, filename, spatial_train=True)

        model = train(lfd_params, model, verbose=True)
        model.save_model()

    if EVAL:
        model = ClassifierBackboneTSM(lfd_params, filename, spatial_train=False)

        train_df = evaluate(lfd_params, model, mode="train")
        train_df["mode"] = ["train"]*len(train_df)
        eval_df = evaluate(lfd_params, model, mode="evaluation")
        eval_df["mode"] = ["evaluation"] * len(eval_df)
        df = train_df + eval_df

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_single_action.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

