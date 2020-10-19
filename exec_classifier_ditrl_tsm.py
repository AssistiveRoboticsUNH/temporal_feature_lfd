import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from run_ditrl_pipeline import train_pipeline, generate_itr_files
from model.classifier_ditrl_tsm import ClassifierDITRLTSM

if __name__ == '__main__':

    lfd_params = default_model_args()  # parse_model_args()

    dir_name = "saved_models/classifier_backbone_tsm"  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    model = ClassifierDITRLTSM(lfd_params, filename, spatial_train=True)
    model = train(lfd_params, model)
    model.save_model()

    model = ClassifierDITRLTSM(lfd_params, filename, spatial_train=False, ditrl_pipeline_train=True)
    model = train_pipeline(lfd_params, model)
    model.save_model()

    generate_itr_files(lfd_params, model, "train")
    generate_itr_files(lfd_params, model, "evaluation")

    model = ClassifierDITRLTSM(lfd_params, filename, spatial_train=False, ditrl_pipeline_train=False, temporal_train=True)
    model = train(lfd_params, model)
    model.save_model()

    df = evaluate(lfd_params, model)

    out_filename = os.path.join(lfd_params.args.output_dir, "output_" + lfd_params.args.save_id + ".csv")
    df.to_csv(out_filename)
    print("Output placed in: " + out_filename)
