import pandas as pd
import os

from enums import suffix_dict, model_dict, Suffix
from parameter_parser import default_model_params

from model.classifier import Classifier
from model.policy_learner import PolicyLearner

from run_classification import train as train_c_iad, evaluate as evaluate_c_iad
from run_classification_gcn import train as train_c_itr, evaluate as evaluate_c_itr

from run_policy_learning import train as train_pl_iad, evaluate_action_trace as evaluate_pl_iad
from run_policy_learning_gcn import train as train_pl_itr, evaluate_action_trace as evaluate_pl_itr

from run_classification import generate_iad_files as generate_iad_files_code
from run_ditrl_pipeline import train_pipeline as train_pipeline_code, generate_itr_files_gcn as generate_itr_files_code


def make_model_name(args, lfd_params, backbone=False):

    # saved backbone models
    save_id = lfd_params.model.save_id
    old_save_dir = os.path.join(lfd_params.base_model_dir, save_id)

    new_save_id = f"{args.app}_{args.suffix}_{args.model}_{args.cur_repeat}"
    new_save_dir = os.path.join(lfd_params.model_save_dir, new_save_id)

    if backbone or not os.path.exists(old_save_dir):
        print("directory ["+old_save_dir+"] does not exist, proceeding anyways")
        return new_save_id

    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)

        from shutil import copy2
        for f in os.listdir(old_save_dir):
            copy2(os.path.join(old_save_dir, f), new_save_dir)

    return new_save_id


def define_model(args, lfd_params, train, app=None, suffix=None, use_bottleneck=False, backbone=False):
    backbone_id = model_dict[args.model]
    filename = make_model_name(args, lfd_params, backbone=backbone)

    if app is None:
        app = args.app

    use_feature_extractor = False
    use_spatial = False
    use_pipeline = False
    use_temporal = False

    train_feature_extractor = False
    train_spatial = False
    train_pipeline = False
    train_temporal = False

    if suffix == Suffix.BACKBONE:
        use_feature_extractor = True
        use_spatial = True
        train_feature_extractor = train
        train_spatial = train
    elif suffix == Suffix.GENERATE_IAD:
        use_feature_extractor = True
        use_spatial = False
    elif suffix == Suffix.PIPELINE:
        use_pipeline = True
        train_pipeline = train
    elif suffix in [Suffix.LINEAR, Suffix.LINEAR_IAD, Suffix.LSTM_IAD, Suffix.LSTM]:
        use_spatial = True
        train_spatial = train
    elif suffix == Suffix.DITRL:
        use_temporal = True
        train_temporal = train
    else:
        print(f"ERROR: execute.py: suffix '{suffix}' not available")
        return None

    # classifier
    if app == 'c' or suffix in [Suffix.PIPELINE, suffix.GENERATE_IAD]:
        return Classifier(lfd_params, filename, backbone_id, suffix,
                          use_feature_extractor=use_feature_extractor, train_feature_extractor=train_feature_extractor,
                          use_bottleneck=backbone,
                          use_spatial=use_spatial, train_spatial=train_spatial,
                          use_pipeline=use_pipeline, train_pipeline=train_pipeline,
                          use_temporal=use_temporal, train_temporal=train_temporal)

    # policy_learner
    return PolicyLearner(lfd_params, filename, backbone_id, suffix,
                         use_feature_extractor=use_feature_extractor, train_feature_extractor=train_feature_extractor,
                         use_bottleneck=backbone,
                         use_spatial=use_spatial, train_spatial=train_spatial,
                         use_pipeline=use_pipeline, train_pipeline=train_pipeline,
                         use_temporal=use_temporal, train_temporal=train_temporal,
                         train_policy=train)


def generate_iad_files(args, lfd_params, model):
    backbone_id = args.model

    for mode in ['train', 'evaluation']:
        generate_iad_files_code(lfd_params, model, mode, backbone=backbone_id)


def generate_itr_files(args, lfd_params, model):
    backbone_id = args.model

    model = train_pipeline_code(lfd_params, model)
    model.save_model()
    for mode in ['train', 'evaluation']:
        generate_itr_files_code(lfd_params, model, mode, backbone=backbone_id)


def train(args, lfd_params, model):
    print("train suffix:", args.suffix)

    if args.app == 'c':
        if args.suffix in ['backbone']:
            return train_c_iad(lfd_params, model, verbose=True, input_dtype="video")
        elif args.suffix in ['linear', 'lstm']:
            return train_c_iad(lfd_params, model, verbose=False, input_dtype="iad")
        elif args.suffix in ['ditrl']:
            return train_c_itr(lfd_params, model, verbose=False, input_dtype="gcn")
    else:
        if args.suffix in ['linear', 'lstm']:
            return train_pl_iad(lfd_params, model, verbose=False, input_dtype="iad")
        elif args.suffix in ['ditrl']:
            return train_pl_itr(lfd_params, model, verbose=False, input_dtype="gcn")
        else:
            print(f"suffix '{args.suffix}' is not intended for use with policy learning")


def evaluate(args, lfd_params, model, mode):
    if args.app == 'c':
        if args.suffix in ['backbone']:
            return evaluate_c_iad(lfd_params, model,  verbose=True, mode=mode, input_dtype="video")
        elif args.suffix in ['linear', 'lstm']:
            return evaluate_c_iad(lfd_params, model,  verbose=False, mode=mode, input_dtype="iad")
        elif args.suffix in ['ditrl']:
            return evaluate_c_itr(lfd_params, model,  verbose=False, mode=mode, input_dtype="gcn")
    else:
        if args.suffix in ['linear', 'lstm']:
            return evaluate_pl_iad(lfd_params, model, verbose=False, mode=mode, input_dtype="iad", ablation=False)
        elif args.suffix in ['ditrl']:
            return evaluate_pl_itr(lfd_params, model, verbose=False, mode=mode, input_dtype="gcn", ablation=False)
        else:
            print(f"suffix '{args.suffix}' is not intended for use with policy learning")


def execute_func(args, lfd_params, cur_repeat, backbone=False):
    suffix = suffix_dict[args.suffix]
    args.cur_repeat = cur_repeat

    # generate files
    if args.generate_files:
        print("Generate Files...")
        if args.suffix not in ['backbone']:
            print("Generate IAD...")
            model = define_model(args, lfd_params, train=False, app='c', suffix=suffix.GENERATE_IAD,
                                 #use_bottleneck=True,
                                 backbone=backbone)
            generate_iad_files(args, lfd_params, model)

            if args.suffix in ['ditrl']:
                print("Generate ITR...")
                model = define_model(args, lfd_params, train=True, app='c', suffix=Suffix.PIPELINE)
                generate_itr_files(args, lfd_params, model)
        print("Done!")

    # train
    if not args.eval_only:
        print("Train Model...")
        model = define_model(args, lfd_params, train=True, suffix=suffix)
        model = train(args, lfd_params, model)
        model.save_model()
        print("Done!")

    # eval
    print("Evaluate Model...")
    model = define_model(args, lfd_params, train=False, suffix=suffix)
    train_df = evaluate(args, lfd_params, model, mode="train")
    eval_df = evaluate(args, lfd_params, model, mode="evaluation")
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
    parser.add_argument('suffix', help='suffix', choices=['backbone', 'linear', 'lstm', 'ditrl'])

    parser.set_defaults(generate_files=False)
    parser.add_argument('--gen', help='generate_files', dest='generate_files', action='store_true')
    parser.set_defaults(eval_only=False)
    parser.add_argument('--eval', help='evaluate only', dest='eval_only', action='store_true')

    parser.add_argument('--frames', help='number of frames', default=64, type=int)
    parser.add_argument('--repeat', help='repeat code runs', default=1, type=int)

    return parser.parse_args()


########


def exec_repeats(args, lfd_params):
    for r in range(args.repeat):
        execute_func(args, lfd_params, r)


if __name__ == '__main__':
    args = parse_exec_args()
    lfd_params = default_model_params()
    lfd_params.set_application("block_construction_timed")
    lfd_params.set_model_params(model_dict[args.model], end_point=-1)
    #lfd_params.set_application("tea_making")
    #lfd_params.epochs = 3

    exec_repeats(args, lfd_params)

