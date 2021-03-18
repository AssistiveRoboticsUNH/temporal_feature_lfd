from enum import Enum


class Suffix(Enum):
    LINEAR = 0
    LSTM = 1
    DITRL = 2
    LINEAR_IAD = 3
    LSTM_IAD = 4
    TCN = 5
    BACKBONE = 100
    PIPELINE = 200  # helper suffix
    GENERATE_IAD = 201  # helper suffix
    NONE = 999



suffix_dict = {"linear": Suffix.LINEAR,
               "lstm": Suffix.LSTM,
               "linear_iad": Suffix.LINEAR_IAD,
               "lstm_iad": Suffix.LSTM_IAD,
               "ditrl": Suffix.DITRL,
               "backbone": Suffix.BACKBONE,
               "tcn": Suffix.TCN}


class Backbone(Enum):
    TSM = 0  # Temporal Shift Module
    VGG = 1  # VGG-16
    WRN = 2  # WideResNet
    I3D = 3  # Inception
    TRN = 4  # Temporal Relation Network


model_dict = {"tsm": Backbone.TSM,
              "vgg": Backbone.VGG,
              "wrn": Backbone.WRN,
              "i3d": Backbone.I3D,
              "trn": Backbone.TRN}