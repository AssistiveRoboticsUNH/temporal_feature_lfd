import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from model.classifier_backbone_i3d import ClassifierBackboneI3D

if __name__ == '__main__':

    lfd_params = default_model_args(epochs=1)  # parse_model_args()

    dir_name = "saved_models/classifier_backbone_i3d"  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    model = ClassifierBackboneI3D(lfd_params, filename, spatial_train=True)

    model = train(lfd_params, model, verbose=True)
    model.save_model()

    df = evaluate(lfd_params, model)

    out_filename = os.path.join(lfd_params.args.output_dir, "output_" + lfd_params.args.save_id + ".csv")
    df.to_csv(out_filename)
    print("Output placed in: " + out_filename)
