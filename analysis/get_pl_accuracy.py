import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

from sklearn.metrics import confusion_matrix, accuracy_score

def get_accuracy_per_action(df, timesteps):
    expected = np.concatenate([df["expected_label_"+str(i)] for i in range(timesteps)])
    predicted = np.concatenate([df["predicted_label_"+str(i)] for i in range(timesteps)])
    #print(pd.DataFrame({"expected": expected, "predicted:": predicted}))
    return accuracy_score(y_true=expected, y_pred=predicted)


def get_accuracy_per_obs(df, timesteps):
    # order by observation

    for i in range(timesteps):
        df["filename_"+str(i)] = df["obs_filename_"+str(i)].str.split('/').str[-1]
        print("exp:", df["expected_label_"+str(i)].dtype, "pred:", df["predicted_label_"+str(i)].dtype)
        df["correct_"+str(i)] = df["expected_label_"+str(i)] == df["predicted_label_"+str(i)]

    df["correct"] = (df["correct_0"] & df["correct_1"] & df["correct_2"]).astype(float)
    df["label"] = df["filename_0"].str.split('_').str[0]

    df = df.groupby("label").mean()
    print(df["correct"])
    #for label in df["label"].unique:
    #    print(label, df[label])

    #expected = np.concatenate([df["expected_label_"+str(i)] for i in range(timesteps)])
    #predicted = np.concatenate([df["predicted_label_"+str(i)] for i in range(timesteps)])

    #return accuracy_score(y_true=expected, y_pred=predicted)


def view_accuracy(df, filename):
    #print(df)
    #print(df["obs_filename_0"])
    df["filename"] = df["obs_filename_0"].str.split('/').str[-1]
    #print(df["filename"])

    df["obs_label"] = df["filename"].str.split('_').str[0]
    #print(df["obs_label"])

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


    #print("len(obs), len(time), len(action):", len(obs), len(time), len(action))

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
    #print(percent_df)

    fig, axs = plt.subplots(8, 3)
    for o in range(len(obs_list)):
        for t in [0, 1, 2]:
            data = percent_df[percent_df["obs"] == obs_list[o]]
            data = data[data["time"] == t]
            data = data["percent"]
            #print(data)
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

    import sys
    model_p = sys.argv[1]
    ditrl = int(sys.argv[2])
    visualize_ablation = int(sys.argv[3])

    model_type = "backbone"
    if ditrl == 1:
        model_type = "iad"
    elif ditrl == 2:
        model_type = "vee"
    elif ditrl == 3:
        model_type = "ditrl"

    if model_p == "tsm":
        save_id = "output_policy_learning_"+model_type+"_tsm"
    elif model_p == "vgg":
        save_id = "output_policy_learning_"+model_type+"_vgg"
    elif model_p == "wrn":
        save_id = "output_policy_learning_"+model_type+"_wrn"
    elif model_p == "r21d":
        save_id = "output_policy_learning_"+model_type+"_r21d"
    elif model_p == "i3d":
        save_id = "output_policy_learning_"+model_type+"_i3d"

    #results_dir = os.path.join("stored_results/lstm/", model_p)
    results_dir = os.path.join("csv_output/")
    dataset_eval = os.path.join(results_dir, save_id+"_action_trace.csv")
    abl_train = os.path.join(results_dir, save_id + "_action_trace_ablation_train.csv")
    abl_eval = os.path.join(results_dir, save_id + "_action_trace_ablation_eval.csv")

    print("dataset_eval:", dataset_eval)
    print("abl_train:", abl_train)
    print("abl_eval:", abl_eval)

    df = pd.read_csv(dataset_eval)
    print("dataset, per_act, eval:", get_accuracy_per_action(df, timesteps=5))
    print("-----")

    df = pd.read_csv(abl_train)
    print("ablation, per_act, train:", get_accuracy_per_action(df, timesteps=3))
    df = pd.read_csv(abl_eval)
    print("ablation, per_act, eval:", get_accuracy_per_action(df, timesteps=3))
    print("-----")

    df = pd.read_csv(abl_train)
    print("ablation, per_obs, train:")
    get_accuracy_per_obs(df, timesteps=3)
    df = pd.read_csv(abl_eval)
    print("\nablation, per_obs, eval:")
    get_accuracy_per_obs(df, timesteps=3)
    print("\nabl_eval:")
    print(pd.DataFrame({"l": df["label"],
                        "e0": df["expected_label_0"], "e1": df["expected_label_1"], "e2": df["expected_label_2"],
                        "p0": df["expected_label_0"], "p1": df["expected_label_1"], "p2": df["expected_label_2"],
                        "c0": df["correct_0"], "c1": df["correct_1"], "c2": df["correct_2"],
                        }))

    if visualize_ablation:
        df = pd.read_csv(abl_train)
        view_accuracy(df, abl_train)

        df = pd.read_csv(abl_eval)
        view_accuracy(df, abl_eval)
