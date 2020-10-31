import os
import pandas as pd
from parameter_parser import parse_model_args, default_model_args
from run_classification import train as train_classification
from run_classification import evaluate as evaluate_classification
from run_ditrl_pipeline import train_pipeline, generate_itr_files
from run_policy_learning import train, evaluate_single_action, evaluate_action_trace
from model.classifier_ditrl_tsm import ClassifierDITRLTSM
from model.policylearner_ditrl_tsm import PolicyLearnerDITRLTSM

TRAIN = True
EVAL = True

def main(save_id, train_p, eval_p):
    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name)  # parse_model_args()

    if train_p:
        '''
        print("Training Spatial Features")
        model = ClassifierDITRLTSM(lfd_params, filename, use_feature_extractor=True, use_spatial=True, use_pipeline=False, use_temporal=False,
                                   spatial_train=True)  # ditrl is true but unused
        model = train_classification(lfd_params, model, input_dtype="video")
        model.save_model()
        
        print("Training Pipeline")
        model = ClassifierDITRLTSM(lfd_params, filename, use_feature_extractor=True, use_spatial=False, use_pipeline=True, use_temporal=False,
                                   spatial_train=False, ditrl_pipeline_train=True)
        model = train_pipeline(lfd_params, model)
        model.save_model()

        #print("model.pipeline.is_training:", model.pipeline.is_training)

        print("Generating ITR Files")
        generate_itr_files(lfd_params, model, "train", backbone="tsm")
        generate_itr_files(lfd_params, model, "evaluation", backbone="tsm")
        '''
        print("Training Policy")

        model = PolicyLearnerDITRLTSM(lfd_params, filename, use_feature_extractor=True, use_spatial=True, use_pipeline=False, use_temporal=True,
                                   spatial_train=False, ditrl_pipeline_train=False, temporal_train=True)
        model = train(lfd_params, model, input_dtype="video", verbose=True)  # make sure to use ITRs
        model.save_model()

    if eval_p:
        '''
        print("Evaluating Model")
        model = ClassifierDITRLTSM(lfd_params, filename, use_feature_extractor=True, use_spatial=True,
                                   use_pipeline=False, use_temporal=False,
                                   spatial_train=False)
        train_df = evaluate_classification(lfd_params, model, mode="train")
        train_df["mode"] = ["train"] * len(train_df)
        eval_df = evaluate_classification(lfd_params, model, mode="evaluation", verbose=True)
        eval_df["mode"] = ["evaluation"] * len(eval_df)
        df = pd.concat([train_df, eval_df])
        df["repeat"] = [save_id] * len(df)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_spatial.csv")
        df.to_csv(out_filename)
        #return df

        print("Output placed in: " + out_filename)

        '''
        model = PolicyLearnerDITRLTSM(lfd_params, filename, use_feature_extractor=True, use_spatial=True,
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

    save_id = "policy_learning_ditrl_tsm_bn16_2"
    main(save_id, TRAIN, EVAL)