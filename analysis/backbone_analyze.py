import pandas as pd
import numpy as np
import os


def locate_files(src_dir, model):

    files = []
    for r, d, f in os.walk(src_dir):
        for filename in f:
            if filename == "results.csv":
                file_path = os.path.join(r, filename)
                #run_name = r.split('/')[-1]
                files.append((r, file_path))
    return files


def organize_data(files):
    df_arr = []

    # organize each file by accuracy
    for run_name, file_path in files:
        df = pd.read_csv(file_path)
        df["run_name"] = [run_name]*len(df)
        df["correct"] = df["expected_label"] == df["predicted_label"]
        df["correct"] = df["correct"].astype(int)

        df = df.groupby(["mode", "expected_label"]).mean().reset_index()
        df = pd.DataFrame({"run_name": [run_name]*len(df["mode"]),
                           "mode": df["mode"],
                           "expected_label": df["expected_label"],
                           "accuracy": df["correct"]})

        df_arr.append(df)

    # collect together and present
    df = pd.concat(df_arr)

    paired_df = {}
    for mode in ["train", "evaluation"]:
        for label, label_n in [('n', 0), ('r', 1), ('rr', 2), ('rrr', 3), ('g', 4), ('gb', 5), ('bg', 6), ('b', 7)]:
            paired_df["run_name"] = df[(df["mode"] == mode) & (df["expected_label"] == label_n)]["run_name"].to_numpy()
            paired_df[mode+'_'+label] = df[(df["mode"] == mode) & (df["expected_label"] == label_n)]["accuracy"].to_numpy()

    df2 = pd.DataFrame(paired_df)
    df2 = df2.drop(columns=["train_rr", "train_rrr", "train_bg", "train_gb",
                            "evaluation_rr", "evaluation_rrr", "evaluation_bg", "evaluation_gb"])
    print(df2)


if __name__ == '__main__':

    src_dir = "."
    model = "i3d"

    files = locate_files(src_dir, model)
    organize_data(files)