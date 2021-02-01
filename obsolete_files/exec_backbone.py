from model.classifier import Classifier
import pandas as pd


def exec_classifier_backbone(args):

    # Train
    if args.eval_only:
        model = Classifier(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial=True,
                           spatial_train=True, use_bottleneck=use_bottleneck)

        model = train(lfd_params, model, verbose=True)
        model.save_model()

    # Evaluate
    model = Classifier(lfd_params, filename, backbone_id, use_feature_extractor=False, use_spatial=True,
                       spatial_train=False, use_bottleneck=use_bottleneck)

    train_df = evaluate(lfd_params, model, mode="train")
    train_df["mode"] = ["train"] * len(train_df)
    eval_df = evaluate(lfd_params, model, mode="evaluation", verbose=True)
    eval_df["mode"] = ["evaluation"] * len(eval_df)
    df = pd.concat([train_df, eval_df])
    df["repeat"] = ["1"] * len(df)

    out_filename = os.path.join(lfd_params.args.output_dir, "output_" + save_id + "_spatial.csv")
    df.to_csv(out_filename)
    print("Output placed in: " + out_filename)

    return 0


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.add_argument('model', help='model_id')
    parser.add_argument('bottleneck', help='bottleneck_size', type=int)

    parser.set_defaults(eval_only=False)
    parser.add_argument('--eval', help='evaluate only', dest='eval_only', action='store_true')

    parser.add_argument('--frames', help='number of frames', default=64, type=int)
    parser.add_argument('--repeat', help='repeat runs', default=1, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_exec_args()

    for i in range(args.repeat):
        exec_classifier_backbone()

