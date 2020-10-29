import os
import pandas as pd
from parameter_parser import parse_model_args, default_model_args
from run_policy_learning import train as train_pl, evaluate_single_action, evaluate_action_trace
from run_classification import train as train_cl, evaluate as evaluate_cl
from model.policylearner_backbone_tsm import PolicyLearnerBackboneTSM
from model.classifier_backbone_tsm import ClassifierBackboneTSM

TRAIN = False
EVAL = True


def main(save_id, train_p, eval_p):

    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name)  # parse_model_args()

    if train_p:
        model = ClassifierBackboneTSM(lfd_params, filename, spatial_train=True)
        model = train_cl(lfd_params, model)
        model.save_model()
        '''
        model = PolicyLearnerBackboneTSM(lfd_params, filename, spatial_train=False, policy_train=True)
        model = train_pl(lfd_params, model, verbose=True)
        model.save_model()
        '''

    if eval_p:
        model = ClassifierBackboneTSM(lfd_params, filename, spatial_train=False)

        train_df = evaluate_cl(lfd_params, model, mode="train")
        train_df["mode"] = ["train"] * len(train_df)
        eval_df = evaluate_cl(lfd_params, model, mode="evaluation", verbose=True)
        eval_df["mode"] = ["evaluation"] * len(eval_df)
        df = pd.concat([train_df, eval_df])
        df["repeat"] = [save_id] * len(df)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_spatial.csv")
        return df
        #df.to_csv(out_filename)
        print("Output placed in: " + out_filename)
        '''
        model = PolicyLearnerBackboneTSM(lfd_params, filename, spatial_train=False, policy_train=False)

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
        '''

if __name__ == '__main__':
    save_id = "policy_learning_backbone_tsm"
    main(save_id, TRAIN, EVAL)