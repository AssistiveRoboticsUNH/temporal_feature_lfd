import pandas as pd
import os

from enums import suffix_dict, model_dict
from parameter_parser import default_model_params

import numpy as np
import torch
from torch_geometric.data import Data

from execute import define_model


def prune_graph(graph, feature_to_prune):
    x = graph.x
    edge_index = graph.edge_index
    edge_attr = graph.edge_attr

    node_feature_label = np.argmax(x, axis=1)
    x = x[np.where(node_feature_label != feature_to_prune)]

    edges_to_remove = set(np.where(edge_index == feature_to_prune)[1].tolist())
    edges_to_keep = set(list(range(edge_index.shape[1]))).difference(edges_to_remove)
    edges_to_keep = list(edges_to_keep)

    edge_index = edge_index[:, edges_to_keep]
    edge_attr = edge_attr[edges_to_keep]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def evaluate_c_itr(lfd_params, model, mode="evaluation", verbose=False):

    # Create DataLoaders

    from datasets.dataset_gcn import DatasetGCN as CustomDataset
    dataset = CustomDataset(lfd_params, lfd_params.application.file_directory, mode, verbose=True,
                            num_segments=lfd_params.input_frames, backbone=lfd_params.model.model_id)
    from datasets.utils_gcn import create_dataloader
    data_loader = create_dataloader(dataset, lfd_params, mode, shuffle=False)

    # put model on GPU
    net = torch.nn.DataParallel(model, device_ids=lfd_params.gpus).cuda()
    net.eval()

    # Get baseline values
    #--------------
    baseline_logits = {}

    print("Obtain Baseline Logits...")
    for i, data_packet in enumerate(data_loader):
        obs, label, filename = data_packet
        expected_label = label.cpu().detach().numpy()[0]

        # compute output
        logits = net(obs)

        baseline_logits[filename[0]] = logits.detach().cpu().numpy()[0, expected_label]

    # Get masked values
    # --------------
    feature_importance_list = []

    for feat in range(lfd_params.model.bottleneck_size):
        print("Obtain Importance for feature ...", feat)
        #feature_importance_total = 0
        feature_importance_by_class = [0] * lfd_params.application.num_labels
        file_count_by_class = [0] * lfd_params.application.num_labels

        for i, data_packet in enumerate(data_loader):
            obs, label, filename = data_packet
            expected_label = label.cpu().detach().numpy()[0]

            # prune obs to remove feature
            prune_graph(obs, feat)

            # compute output
            logits = net(obs)
            new_logits_value = logits[0, expected_label]
            feature_importance = (baseline_logits[filename[0]] - new_logits_value) / baseline_logits[filename[0]]
            feature_importance = feature_importance.detach().cpu().numpy()

            file_count_by_class[expected_label] += 1
            feature_importance_by_class[expected_label] += feature_importance
            #feature_importance_total += feature_importance

        feature_importance_by_class = np.array(feature_importance_by_class) / np.array(file_count_by_class)
        #feature_importance_total /= len(data_loader)

        #importance_value = feature_importance_total.detach().cpu().numpy()
        #print("importance_value:", feature_importance_by_class)
        feature_importance_list.append(feature_importance_by_class)

    feature_importance_list = np.array(feature_importance_list)
    #print("feature_importance_list:", feature_importance_list.shape)

    # return Pandas dataframe
    importance_pd = {"feature": np.arange(len(feature_importance_list))}
    for class_label in range(lfd_params.application.num_labels):
        importance_pd["importance_label_"+str(class_label)] = feature_importance_list[:, class_label]
    importance_pd["importance_total"] = np.sum(feature_importance_list, axis=1)

    #for k, v in importance_pd.items():
    #    print(k, v.shape)

    return pd.DataFrame(importance_pd)


########


def evaluate(args, lfd_params, model, mode):
    return evaluate_c_itr(lfd_params, model,  verbose=False, mode=mode)


def execute_func(args, lfd_params, cur_repeat):
    suffix = suffix_dict[args.suffix]
    args.cur_repeat = cur_repeat

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
    print(df)
    save_path = os.path.join(model.filename, "importance_temporal.csv")
    print("save_path:", save_path)
    df.to_csv(save_path)

    return df


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Perform node ranking using erasure search: https://arxiv.org/pdf/1612.08220.pdf')

    parser.set_defaults(app='c')
    parser.add_argument('model', help='model_id', choices=model_dict.keys())
    parser.set_defaults(suffix='ditrl')

    parser.set_defaults(generate_files=False)
    parser.add_argument('--gen', help='generate_files', dest='generate_files', action='store_true')
    parser.set_defaults(eval_only=True)
    #parser.add_argument('--eval', help='evaluate only', dest='eval_only', action='store_true')

    parser.add_argument('--frames', help='number of frames', default=64, type=int)
    parser.set_defaults(repeat=1)

    return parser.parse_args()


########


def exec_repeats(args, lfd_params):
    for r in range(args.repeat):
        execute_func(args, lfd_params, r)


if __name__ == '__main__':
    args = parse_exec_args()
    lfd_params = default_model_params()
    lfd_params.set_model_params(model_dict[args.model], end_point=-1)
    lfd_params.set_application("tea_making")

    exec_repeats(args, lfd_params)
