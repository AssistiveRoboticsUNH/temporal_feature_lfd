import os
import pandas as pd
from parameter_parser import parse_model_args, default_model_args
from run_classification_gcn import train, evaluate
from run_ditrl_pipeline import train_pipeline, generate_itr_files, generate_itr_files_gcn
from model.classifier_ditrl_tsm_gcn import ClassifierDITRLTSM as ClassifierDITRLTSMGCN
from model.classifier_ditrl_tsm import ClassifierDITRLTSM


TRAIN = True
EVAL = True

def main(save_id, train_p, eval_p):
    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name, num_segments=20)

    if train_p:
        '''
        print("Training Spatial Features")
        model = ClassifierDITRLTSM(lfd_params, filename, use_feature_extractor=True, use_spatial=True, use_pipeline=False, use_temporal=False,
                                   spatial_train=True)  # ditrl is true but unused
        model = train(lfd_params, model, input_dtype="video", verbose=True)
        model.save_model()
        '''


        print("Training Pipeline")
        model = ClassifierDITRLTSM(lfd_params, filename, use_feature_extractor=True, use_spatial=False, use_pipeline=True, use_temporal=False,
                                   spatial_train=False, ditrl_pipeline_train=True, use_gcn=True)
        model = train_pipeline(lfd_params, model)
        model.save_model()

        #print("model.pipeline.is_training:", model.pipeline.is_training)
        
        print("Generating GCN Files")
        generate_itr_files_gcn(lfd_params, model, "train", max=500)
        generate_itr_files_gcn(lfd_params, model, "evaluation", max=500)

        model = ClassifierDITRLTSMGCN(lfd_params, filename, use_feature_extractor=False, use_spatial=False,
                                   use_pipeline=False, use_temporal=True,
                                   spatial_train=False, ditrl_pipeline_train=False, temporal_train=True, use_gcn=True)
        model = train(lfd_params, model, input_dtype="gcn", verbose=True)  # make sure to use ITRs
        model.save_model()

    if eval_p:
        print("Evaluating Model")
        model = ClassifierDITRLTSMGCN(lfd_params, filename, use_feature_extractor=False, use_spatial=False,
                                   use_pipeline=False, use_temporal=True,
                                   spatial_train=False, ditrl_pipeline_train=False, temporal_train=False, use_gcn=True)

        train_df = evaluate(lfd_params, model, mode="train", input_dtype="gcn")
        train_df["mode"] = ["train"] * len(train_df)
        eval_df = evaluate(lfd_params, model, mode="evaluation", verbose=True, input_dtype="gcn")
        eval_df["mode"] = ["evaluation"] * len(eval_df)
        df = pd.concat([train_df, eval_df])
        df["repeat"] = [save_id] * len(df)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + ".csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

if __name__ == '__main__':

    save_id = "policy_learning_ditrl_tsm_bn16_2" #classifier_ditrl_tsm
    main(save_id, TRAIN, EVAL)
