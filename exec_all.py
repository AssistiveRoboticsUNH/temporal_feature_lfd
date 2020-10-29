
from exec_policy_learning_backbone_tsm import main as main_bb_tsm
from exec_policy_learning_backbone_i3d import main as main_bb_i3d
from exec_policy_learning_ditrl_tsm import main as main_ditrl_tsm
from exec_policy_learning_ditrl_tsm import main as main_ditrl_i3d

if __name__ == '__main__':
    for i in range(5):
        #main_bb_tsm("policy_learning_backbone_tsm_" + str(i), True, True)
        #main_bb_i3d("policy_learning_backbone_i3d_" + str(i), True, True)
        #main_ditrl_tsm("policy_learning_ditrl_tsm2_" + str(i), True, True)
        main_ditrl_i3d("policy_learning_ditrl_i3d_" + str(i), True, True)
