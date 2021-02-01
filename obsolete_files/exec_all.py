import pandas as pd
import os
import sys

from obsolete_files.exec_classifier_bottleneck import main


if __name__ == '__main__':
    out_df = None
    for i in range(1):

        train = True
        eval = True
        model = sys.argv[1]
        save_id = "classifier_bottleneck_" + model + str(i)

        df = main(save_id, train, eval, model, return_eval=True)
        df["repeat"] = [str(i)]*len(df)

        if out_df is None:
            out_df = df
        else:
            out_df = pd.concat((out_df, df))

    out_filename = os.path.join("csv_output/classifier_bottleneck_"+model+".csv")
    out_df.to_csv(out_filename)
