import pandas as pd
import os
from exec_policy_learning_backbone_tsm2 import main as main_bb_tsm
from exec_policy_learning_backbone_i3d import main as main_bb_i3d
from exec_policy_learning_ditrl_tsm import main as main_ditrl_tsm
from exec_policy_learning_ditrl_i3d import main as main_ditrl_i3d

if __name__ == '__main__':
    out_df = None
    for i in range(3):
        #main_bb_tsm(policy_learning_backbone_tsm_" + str(i), True, True)
        #main_bb_i3d("policy_learning_backbone_i3d_" + str(i), True, True)
        #main_ditrl_tsm("policy_learning_ditrl_tsm2_" + str(i), True, True)
        #main_ditrl_i3d("policy_learning_ditrl_i3d_" + str(i), True, True)

        #df = main_ditrl_tsm("policy_learning_ditrl_tsm_bn16_" + str(i), True, True)

        df = main_ditrl_i3d("policy_learning_ditrl_i3d_bn16_" + str(i), False, True)
        if out_df is None:
            out_df = df
        else:
            out_df = pd.concat((out_df, df))

    out_filename = os.path.join("csv_output", "output_" + "i3d_runbn16" + "_spatial.csv")
    out_df.to_csv(out_filename)
