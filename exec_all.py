import pandas as pd
import os


def main():
    pass


if __name__ == '__main__':
    out_df = None
    for i in range(3):


        df = main("policy_learning_ditrl_i3d_bn16_" + str(i), False, True)
        if out_df is None:
            out_df = df
        else:
            out_df = pd.concat((out_df, df))

    out_filename = os.path.join("csv_output", "output_" + "i3d_runbn16" + "_spatial.csv")
    out_df.to_csv(out_filename)
