import os
from parameter_parser import parse_model_args, default_model_args
from run_classification import train, evaluate
from model.classifier_backbone_r21d import ClassifierBackboneR21D

TRAIN = False
EVAL = True

def main(save_id, train_p, eval_p):

    lfd_params = default_model_args(epochs=1, num_segments=64)  # parse_model_args()

    dir_name = "saved_models/"+save_id  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    model = ClassifierBackboneR21D(lfd_params, filename, spatial_train=True)

    if train_p:
        model = train(lfd_params, model, verbose=True, dense_sample=True)
        model.save_model()
    if eval_p:
        model = train(lfd_params, model, verbose=True, dense_sample=False)
        df = evaluate(lfd_params, model, dense_sample=True)

        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + ".csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

if __name__ == '__main__':
    save_id = "classifier_backbone_i3d"
    main(save_id, TRAIN, EVAL)
