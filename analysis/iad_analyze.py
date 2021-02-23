import numpy as np
from execute import define_model
from enums import Suffix

def view_iad(model, iad_filename):
    # open IAD
    iad = np.load(iad_filename)

    # using model determine min and max values
    model.pipeline

    # scale values
    # scale and format image as png
    # save
    pass

def view_events(iad):
    pass



def interrogate_files(stuff, func):
    # get list of files and pass through function
    pass



def execute_func(args, lfd_params, cur_repeat, backbone=False):

    # find values
    num_features = 0

    global_min_values = np.zeros(num_features)
    global_max_values = np.zeros(num_features)
    global_avg_values = np.zeros(num_features)
    global_cnt_values = np.zeros(num_features)

    for file in train_files:
        min_values =
        max_values =
        avg_values =
        cnt_values =

        # update globals



    # generate images
    for dataset_files in [train_files, evaluation_files]:
        for file in dataset_files:
            generate_iad_png(file, min_values, max_values)
            generate_event_png(file, avg_values)



if __name__ == '__main__':
