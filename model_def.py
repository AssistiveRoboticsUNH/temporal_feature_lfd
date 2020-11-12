
def define_model(model_p):
    assert model_p in ["tsm", "i3d", "r21d", "eco", "pan", "vgg", "wrn"], \
        "ERROR: model_df.py: model_p (" + model_p + ") not valid"
    num_segments = None
    bottleneck_size = None
    dense_sample = None
    dense_rate = None

    if model_p == "tsm":
        num_segments = 16
        bottleneck_size = 16
        dense_sample = False
        dense_rate = 0
    elif model_p == "i3d":
        num_segments = 64
        bottleneck_size = 8
        dense_sample = True
        dense_rate = 12
    elif model_p == "r21d":
        num_segments = 64
        bottleneck_size = 8
        dense_sample = True
        dense_rate = 1
    elif model_p == "eco":
        num_segments = 16
        bottleneck_size = 16
        dense_sample = False
        dense_rate = 0
    elif model_p == "vgg":
        num_segments = 16#32 for c01
        bottleneck_size = 16
        dense_sample = False
        dense_rate = 0
    elif model_p == "wrn":
        num_segments = 16
        bottleneck_size = 16
        dense_sample = False
        dense_rate = 0

    return {"num_segments": num_segments,
            "bottleneck_size": bottleneck_size,
            "dense_sample": dense_sample,
            "dense_rate": dense_rate}
