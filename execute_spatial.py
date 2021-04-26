import pandas as pd
import os

from enums import suffix_dict, model_dict, Suffix, Format
from parameter_parser import default_model_params, APPLICATION_NAMES

from model.classifier import Classifier
from model.policy_learner import PolicyLearner

from run_classification import train as train_c_iad, evaluate as evaluate_c_iad
from run_classification_gcn import train as train_c_itr, evaluate as evaluate_c_itr

from run_policy_learning import train as train_pl_iad, evaluate_action_trace as evaluate_pl_iad
from run_policy_learning_gcn import train as train_pl_itr, evaluate_action_trace as evaluate_pl_itr

from run_classification import generate_iad_files as generate_iad_files_code
from run_ditrl_pipeline import train_pipeline as train_pipeline_code, generate_itr_files_gcn as generate_itr_files_code

def define_model(args, lfd_params, train, app=None, suffix=None, use_bottleneck=False, backbone=False):
    backbone_id = model_dict[args.model]

    if backbone_id == "tsm":
        from model.backbone_model.tsm.ops.models import TSN as TSM
        output_size = 172
        input_frames = 8
        return TSM(output_size, lfd_params.input_frames,  # num_classes, num_segments
                     'RGB',
                     base_model='resnet101',
                     consensus_type='avg',
                     dropout=0.5,
                     img_feature_dim=256,
                     partial_bn=True,
                     pretrain='imagenet',

                     is_shift=False,
                     shift_div=8,
                     shift_place='blockres',

                     new_length=None,
                     before_softmax=True,

                     fc_lr5=False,
                     temporal_pool=False,
                     non_local=False)


def evaluate(args, lfd_params, model, mode, backbone_type="video"):
    if args.app == 'c':
        if args.suffix in ['backbone']:
            return evaluate_c_iad(lfd_params, model,  verbose=True, mode=mode, input_dtype=backbone_type,
                                  overwrite_path=os.path.join(lfd_params.application.file_directory, "iad_src"))
        elif args.suffix in ['linear', 'lstm', 'tcn']:
            return evaluate_c_iad(lfd_params, model,  verbose=False, mode=mode, input_dtype="iad")
        elif args.suffix in ['ditrl']:
            return evaluate_c_itr(lfd_params, model,  verbose=False, mode=mode, input_dtype="gcn")
    else:
        if args.suffix in ['linear', 'lstm', 'tcn']:
            return evaluate_pl_iad(lfd_params, model, verbose=False, mode=mode, input_dtype="iad", ablation=False)
        elif args.suffix in ['ditrl']:
            return evaluate_pl_itr(lfd_params, model, verbose=False, mode=mode, input_dtype="gcn", ablation=False)
        else:
            print(f"suffix '{args.suffix}' is not intended for use with policy learning")


def execute_func(args, lfd_params, cur_repeat, backbone=False):
    suffix = suffix_dict[args.suffix]
    args.cur_repeat = cur_repeat
    backbone_type = "video" if lfd_params.application.format is not Format.IAD else "iad"

    # eval
    print("Evaluate Model...")
    model = define_model(args, lfd_params, train=False, suffix=suffix, backbone=backbone)
    train_df = evaluate(args, lfd_params, model, backbone_type=backbone_type, mode="train")
    eval_df = evaluate(args, lfd_params, model, backbone_type=backbone_type, mode="evaluation")
    print("Done!")

    # generate output
    train_df["mode"] = ["train"] * len(train_df)
    eval_df["mode"] = ["evaluation"] * len(eval_df)

    df = pd.concat([train_df, eval_df])
    save_path = os.path.join(model.filename, "results.csv")
    print("save_path:", save_path)
    df.to_csv(save_path)

    return df


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.add_argument('app', help='classifier(c)/policy_learner(pl)', choices=['c', 'pl'])
    parser.add_argument('model', help='model_id', choices=model_dict.keys())
    parser.add_argument('suffix', help='suffix', choices=['backbone', 'linear', 'lstm', 'tcn', 'ditrl'])

    parser.add_argument('--frames', help='number of frames', default=8, type=int)
    parser.add_argument('--repeat', help='repeat code runs', default=1, type=int)
    parser.add_argument('--application', help='application', default=APPLICATION_NAMES[0], choices=APPLICATION_NAMES)

    return parser.parse_args()


########

if __name__ == '__main__':
    args = parse_exec_args()
    lfd_params = default_model_params()
    lfd_params.set_application(args.application)
    lfd_params.set_model_params(model_dict[args.model], end_point=-1)
    lfd_params.input_frames = args.frames

    execute_func(args, lfd_params)

