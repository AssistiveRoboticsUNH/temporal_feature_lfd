
def define_model(model_p):
    assert model_p in ["tsm", "i3d", "r21d", "eco", "pan", "vgg", "wrn", "trn"], \
        "ERROR: model_df.py: model_p (" + model_p + ") not valid"
    num_segments = None
    bottleneck_size = None
    dense_sample = None
    dense_rate = None

    fine_segments = 64

    if model_p == "tsm":
        num_segments = fine_segments
        iad_frames = fine_segments
        original_size = 2048
        bottleneck_size = 16
        dense_sample = False
        dense_rate = 0
    elif model_p == "trn":
        num_segments = 64#fine_segments
        iad_frames = 64#fine_segments
        original_size = 2048
        bottleneck_size = 16
        dense_sample = False
        dense_rate = 0
    elif model_p == "i3d":
        num_segments = 64
        end_point = 5

        original_size_array = [64, 192, 256, 832, 1024, 1024]
        iad_frames_array = [32, 32, 32, 16, 8, 8]
        original_size = original_size_array[end_point] #50176  #1024
        iad_frames = iad_frames_array[end_point] #8
        bottleneck_size = 8
        dense_sample = False
        # end_point modification in feature_extractor.py
        #dense_sample = True
        #dense_rate = 12
    elif model_p == "r21d":
        num_segments = 64
        iad_frames = 1
        original_size = 512
        bottleneck_size = 8
        dense_sample = False
        #dense_sample = True
        #dense_rate = 1
    elif model_p == "vgg":
        num_segments = fine_segments
        iad_frames = fine_segments
        original_size = 512#25088 # 512
        bottleneck_size = 32
        dense_sample = False
        dense_rate = 0
    elif model_p == "wrn":
        num_segments = fine_segments
        iad_frames = fine_segments
        original_size = 2048
        bottleneck_size = 16
        dense_sample = False
        dense_rate = 0

    '''
     elif model_p == "eco":
         num_segments = fine_segments
         iad_frames = fine_segments
         original_size = None
         bottleneck_size = 16
         dense_sample = False
         dense_rate = 0
     '''

    return {"num_segments": num_segments,
            "iad_frames": iad_frames,
            "original_size": original_size,
            "bottleneck_size": bottleneck_size,
            "dense_sample": dense_sample,
            "dense_rate": dense_rate}
