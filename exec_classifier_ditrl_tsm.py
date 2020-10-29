import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from run_ditrl_pipeline import train_pipeline, generate_itr_files
from model.classifier_ditrl_tsm import ClassifierDITRLTSM

TRAIN = True
EVAL = True

def main(save_id, train_p, eval_p):
    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name)

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
                                   spatial_train=False, ditrl_pipeline_train=True)
        model = train_pipeline(lfd_params, model)
        model.save_model()

        #print("model.pipeline.is_training:", model.pipeline.is_training)

        print("Generating ITR Files")
        generate_itr_files(lfd_params, model, "train")
        generate_itr_files(lfd_params, model, "evaluation")

        model = ClassifierDITRLTSM(lfd_params, filename, use_feature_extractor=False, use_spatial=False,
                                   use_pipeline=False, use_temporal=True,
                                   spatial_train=False, ditrl_pipeline_train=False, temporal_train=True)
        model = train(lfd_params, model, input_dtype="itr", verbose=True)  # make sure to use ITRs
        model.save_model()

    if eval_p:
        print("Evaluating Model")
        model = ClassifierDITRLTSM(lfd_params, filename, use_feature_extractor=False, use_spatial=False,
                                   use_pipeline=False, use_temporal=True,
                                   spatial_train=False, ditrl_pipeline_train=False, temporal_train=False)

        df = evaluate(lfd_params, model)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + lfd_params.args.save_id + ".csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

if __name__ == '__main__':

    save_id = "policy_learning_ditrl_tsm_bn16_2" #classifier_ditrl_tsm
    main(save_id, TRAIN, EVAL)
