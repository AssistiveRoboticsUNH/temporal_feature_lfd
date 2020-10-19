import os
from parameter_parser import parse_model_args, default_model_args
from run_policy_learning import train, evaluate_single_action, evaluate_action_trace
from model.policylearner_backbone_tsm import PolicyLearnerBackboneTSM

if __name__ == '__main__':

    lfd_params = default_model_args(epochs=1)  # parse_model_args()

    dir_name = "saved_models/policy_learning_backbone_tsm"  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    model = PolicyLearnerBackboneTSM(lfd_params, filename, spatial_train=True, policy_train=True)

    train(lfd_params, model)

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


