import os

from exec_policy_learning_lstm import main as main_bb
from exec_policy_learning_ditrl import main as main_ditrl


def make_model_name(model_p, save_id, ext):
    new_save_id = "policy_learning_"+ext+"_" + model_p
    old_save_dir = os.path.join("base_models", save_id)
    new_save_dir = os.path.join("saved_models", new_save_id)
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)

        from shutil import copy2

        for f in os.listdir(old_save_dir):
            copy2(os.path.join(old_save_dir, f), new_save_dir)
    return new_save_id


if __name__ == '__main__':
    import sys

    model_p = sys.argv[1]
    FULL = int(sys.argv[2])

    save_id = ""
    if model_p == "tsm":
        save_id = "classifier_bottleneck_tsm3"
    elif model_p == "vgg":
        save_id = "classifier_bottleneck_vgg0"
    elif model_p == "wrn":
        save_id = "classifier_bottleneck_wrn1"
    elif model_p == "r21d":
        save_id = "classifier_bottleneck_r21d0"
    elif model_p == "i3d":
        save_id = "classifier_bottleneck_i3d0"


    new_save_id = make_model_name(model_p, save_id, "backbone")
    main_bb(new_save_id, gen_p=True, train_p=True, eval_p=True, backbone_id=model_p, use_bottleneck=False)   # backbone

    new_save_id = make_model_name(model_p, save_id, "iad")
    main_bb(new_save_id, gen_p=True, train_p=True, eval_p=True, backbone_id=model_p, use_bottleneck=True)   # iad

    new_save_id = make_model_name(model_p, save_id, "ditrl")
    #main_ditrl(new_save_id, gen_itr=True, gen_vee=True, train_p=True, eval_p=True, backbone_id=model_p)  # ditrl
    main_ditrl(new_save_id, gen_itr=True, gen_vee=True, train_p=False, eval_p=False, backbone_id=model_p)  # ditrl
    
    new_save_id = make_model_name(model_p, save_id, "vee")
    main_bb(new_save_id, gen_p=False, train_p=True, eval_p=True, backbone_id=model_p, use_bottleneck=True)  # threshold

    print("done")