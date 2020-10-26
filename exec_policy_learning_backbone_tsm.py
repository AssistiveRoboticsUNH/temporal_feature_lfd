import os
from parameter_parser import parse_model_args, default_model_args
from run_policy_learning import train, evaluate_single_action, evaluate_action_trace
from model.policylearner_backbone_tsm import PolicyLearnerBackboneTSM

TRAIN = True
EVAL = False

if __name__ == '__main__':

    save_id = "policy_learning_backbone_tsm"
    dir_name = os.path.join("saved_models", save_id)  # lfd_params
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    filename = os.path.join(dir_name, "model")

    lfd_params = default_model_args(save_id=save_id, log_dir=dir_name, verbose=True)  # parse_model_args()

    if TRAIN:
        model = PolicyLearnerBackboneTSM(lfd_params, filename, spatial_train=True, policy_train=True)

        model = train(lfd_params, model)
        model.save_model()

    if EVAL:
        model = PolicyLearnerBackboneTSM(lfd_params, filename, spatial_train=False, policy_train=False)

        df = evaluate_single_action(lfd_params, model)
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_single_action.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

        df = evaluate_action_trace(lfd_params, model)
        out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_action_trace.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)

        df = evaluate_action_trace(lfd_params, model, ablation=True)
        out_filename = os.path.join(lfd_params.args.output_dir,
                                    "output_" + save_id + "_action_trace_ablation.csv")
        df.to_csv(out_filename)
        print("Output placed in: " + out_filename)


