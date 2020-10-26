import os
from parameter_parser import parse_model_args, default_model_args
from run_policy_learning import train, evaluate_single_action, evaluate_action_trace
from model.policylearner_backbone_i3d import PolicyLearnerBackboneI3D

if __name__ == '__main__':

    lfd_params = default_model_args(num_segments=64)  # parse_model_args()

    dir_name = "saved_models/policy_learning_backbone_i3d"  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    model = PolicyLearnerBackboneI3D(lfd_params, filename, spatial_train=True, policy_train=True)

    model = train(lfd_params, model)
    model.save_model()

    '''
    df = evaluate_single_action(lfd_params, model)
    out_filename = os.path.join(lfd_params.args.output_dir, "output_" + lfd_params.args.save_id + "_single_action.csv")
    df.to_csv(out_filename)
    print("Output placed in: " + out_filename)
    '''

    df = evaluate_action_trace(lfd_params, model)
    out_filename = os.path.join(lfd_params.args.output_dir, "output_" + lfd_params.args.save_id + "_action_trace.csv")
    df.to_csv(out_filename)
    print("Output placed in: " + out_filename)

    df = evaluate_action_trace(lfd_params, model, ablation=True)

    out_filename = os.path.join(lfd_params.args.output_dir,
                                "output_" + lfd_params.args.save_id + "_action_trace_ablation.csv")
    df.to_csv(out_filename)
    print("Output placed in: " + out_filename)