import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from run_ditrl_pipeline import train_pipeline, generate_itr_files
from model.classifier_ditrl_i3d import ClassifierDITRLI3D

if __name__ == '__main__':

    lfd_params = default_model_args(epochs=1, num_segments=64)  # parse_model_args()

    dir_name = "saved_models/classifier_ditrl_i3d"  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    print("Training Spatial Features")
    model = ClassifierDITRLI3D(lfd_params, filename, use_feature_extractor=True, use_spatial=True, use_pipeline=False, use_temporal=False,
                               spatial_train=True)  # ditrl is true but unused
    model = train(lfd_params, model, input_dtype="video", verbose=True)
    model.save_model()

    print("Training Pipeline")
    model = ClassifierDITRLI3D(lfd_params, filename, use_feature_extractor=True, use_spatial=False, use_pipeline=True, use_temporal=False,
                               spatial_train=False, ditrl_pipeline_train=True)
    model = train_pipeline(lfd_params, model)
    model.save_model()

    #print("model.pipeline.is_training:", model.pipeline.is_training)

    print("Generating ITR Files")
    generate_itr_files(lfd_params, model, "train")
    generate_itr_files(lfd_params, model, "evaluation")

    print("Evaluating Model")
    model = ClassifierDITRLI3D(lfd_params, filename, use_feature_extractor=False, use_spatial=False, use_pipeline=False, use_temporal=True,
                               spatial_train=False, ditrl_pipeline_train=False, temporal_train=True)
    model = train(lfd_params, model, input_dtype="itr", verbose=True)  # make sure to use ITRs
    model.save_model()

    df = evaluate(lfd_params, model)

    out_filename = os.path.join(lfd_params.args.output_dir, "output_" + lfd_params.args.save_id + ".csv")
    df.to_csv(out_filename)
    print("Output placed in: " + out_filename)