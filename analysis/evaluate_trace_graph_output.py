import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse
import os

from sklearn.metrics import confusion_matrix, accuracy_score


def get_accuracy(df):
    expected = df["expected_label"]
    observed = df["predicted_label"]

    return accuracy_score(y_true = expected, y_pred = observed )


def get_accuracy_policy_learning(df, timesteps):
	#print("columns:", df.columns)
	#timesteps = int(len(df.columns)/3)
	#print("timesteps:", timesteps)

	expected = np.concatenate([df["expected_label_"+str(i)] for i in range(timesteps)])
	predicted = np.concatenate([df["predicted_label_"+str(i)] for i in range(timesteps)])

	print(expected)
	print(predicted)


	return accuracy_score(y_true=expected, y_pred=predicted )

def view_accuracy(df, filename):
    print(df)
    print(df["obs_filename_0"])
    df["filename"] = df["obs_filename_0"].str.split('/').str[-1]
    print(df["filename"])

    df["obs_label"] = df["filename"].str.split('_').str[0]
    print(df["obs_label"])

    obs_list = ['n', 'r', 'rr', 'rrr', 'g', 'gb', 'bg', 'b']

    obs = []
    time = []
    action = []
    for i in range(len(df)):
        row = df.iloc[i]
        obs.extend([row["obs_label"]]*3)
        time.extend([0, 1, 2])
        '''
        action.append(row["expected_label_0"])
        action.append(row["expected_label_1"])
        action.append(row["expected_label_2"])
        '''
        action.append(row["predicted_label_0"])
        action.append(row["predicted_label_1"])
        action.append(row["predicted_label_2"])


    print("len(obs), len(time), len(action):", len(obs), len(time), len(action))

    new_df = pd.DataFrame({"obs": obs, "time": time, "action": action}, dtype="category")

    count_matrix = np.zeros((len(obs_list), 3, 4))
    for i in range(len(new_df)):
        row = new_df.iloc[i]

        o = obs_list.index(row["obs"])
        t = row["time"]
        a = row["action"]

        count_matrix[o, t, a] += 1

    obs = []
    time = []
    action = []
    percent = []

    for o in range(len(obs_list)):
        for t in [0,1,2]:
            for a in [0,1,2,3]:
                obs.append(obs_list[o])
                time.append(t)
                action.append(a)
                percent.append(float(count_matrix[o, t, a]) / np.sum(count_matrix[o, t]) )

    percent_df = pd.DataFrame({"obs": obs, "time": time, "action": action, "percent": percent})
    print(percent_df)
    """
    '''
    new_df = pd.DataFrame({"obs": df["obs_label"],
                           "predicted_label_0": df["predicted_label_0"],
                           "predicted_label_1": df["predicted_label_1"],
                           "predicted_label_2": df["predicted_label_2"]})
    '''
    new_df = pd.DataFrame({"obs": df["obs_label"],
                           "expected_label_0": df["expected_label_0"],
                           "expected_label_1": df["expected_label_1"],
                           "expected_label_2": df["expected_label_2"]}, dtype="category")
    """
    #new_df = new_df[new_df["obs"] == 'rrr']["expected_label_0"]
    #new_df = new_df[new_df["obs"] == "rrr"]
    #print(new_df.groupby(["obs", "time", "action"]).count())
    #print(new_df.groupby(["obs", "time"]).count())
    #print(new_df.groupby("obs").mean())

    fig, axs = plt.subplots(8, 3)
    for o in range(len(obs_list)):
        for t in [0, 1, 2]:
            data = percent_df[percent_df["obs"] == obs_list[o]]
            data = data[data["time"] == t]
            data = data["percent"]
            print(data)
            #pd.DataFrame(data).plot.bar()
            axs[o, t].bar(["N", "R", "G", "B"], data, color=["black", "red", "green", "blue"])

            axs[o, t].set_ylim(0, 1.0)
            #if t != 0:
            axs[o, t].get_yaxis().set_ticks([])
            if o != len(obs_list)-1:
                axs[o, t].get_xaxis().set_ticks([])

    #percent_df.plot.bar()
    #plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description='Generate IADs from input files')
    parser.add_argument('input_file', help='the checkpoint file to use with the model')
    parser.add_argument('--fig_dir', default="analysis/fig", help='the checkpoint file to use with the model')
    args = parser.parse_args()

    src_filename = args.input_file.split("/")[-1][:-4]

    fig_dir = os.path.join(args.fig_dir, src_filename)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    '''

    '''
    input_file = "csv_output/output_policy_learning_ditrl_i3d_action_trace_ablation.csv"

    #df = pd.read_csv(args.input_file)
    df = pd.read_csv(input_file)

    #acc = get_accuracy(df)
    view_accuracy(df)
    # viz_confusion_matrix(df, os.path.join(fig_dir, "cm.png"))

    #print("filename: ", src_filename)
    #print("accuracy: ", acc)
    '''

    '''
    input_files = ["csv_output/output_policy_learning_backbone_i3d_action_trace_ablation.csv",
                   "csv_output/output_policy_learning_backbone_tsm_action_trace_ablation.csv",
                   "csv_output/output_policy_learning_ditrl_i3d_action_trace_ablation.csv",
                   "csv_output/output_policy_learning_ditrl_tsm_action_trace_ablation.csv"]
    '''
    import sys
    input_files = [sys.argv[1]]
    for f in input_files:
        df = pd.read_csv(f)
        view_accuracy(df, f)

    print("acc:", get_accuracy_policy_learning(df, timesteps=3))

