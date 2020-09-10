import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    # load spatial data
    spatial_df = pd.read_csv(sys.argv[1])
    spatial_df["model"] = ["spatial"] * len(spatial_df)
    spatial_df["correct"] = spatial_df["expected_action"] == spatial_df["observed_action"]
    spatial_df = spatial_df.groupby(["bottleneck"]).mean().reset_index()

    # load temporal data
    temporal_df = pd.read_csv(sys.argv[2])
    temporal_df["model"] = ["temporal"] * len(temporal_df)
    temporal_df["correct"] = temporal_df["expected_action"] == temporal_df["observed_action"]
    temporal_df = temporal_df.groupby(["bottleneck"]).mean().reset_index()

    # df = spatial_df.append(temporal_df)
    labels = spatial_df["bottleneck"].to_numpy()
    spatial_data = spatial_df["correct"].to_numpy()
    temporal_data = temporal_df["correct"].to_numpy()
    df = pd.DataFrame({"spatial": spatial_data,
                       "temporal": temporal_data}, index=labels)

    df.plot.bar()
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Bottleneck Size")

    plt.tight_layout()

    # save plt
    # plt.show()
    plt.savefig("analysis/fig/accuracy.png")



