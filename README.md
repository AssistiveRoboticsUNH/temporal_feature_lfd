# Temporal Feature LfD: D-ITR-L

## Overview

We release the PyTorch code of Deep Interval Temporal Relationship Learner (D-ITR-L)

## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#dataset)
- [Code](#code)
- [Pretrained Models](#pretrained-models)
  * [Kinetics-400](#kinetics-400)
    + [Dense Sample](#dense-sample)
    + [Unifrom Sampling](#unifrom-sampling)
  * [Something-Something](#something-something)
    + [Something-Something-V1](#something-something-v1)
    + [Something-Something-V2](#something-something-v2)
- [Testing](#testing)
- [Training](#training)
- [Live Demo on NVIDIA Jetson Nano](#live-demo-on-nvidia-jetson-nano)

## Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.7 or higher
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [openCV](hhttps://pypi.org/project/opencv-python/) 4.5 or later

## Dataset

We have provided the Block Stacking dataset discussed in our paper here: [insert link].
The dataset is split into two forms: a timed method that tries toa dhere to a specific duration schedule.
And an un-timed dataset that is more variable.

## Code

Our code is applied in two steps: generation of backbone CNN models, and creation of temporal features.

### Backbone CNN

```bash
# test TSN
python3.6 execute_backbone.py --help
```

Trained models will be placed in folders of the type "saved_model_(application)\_(bottleneck_size)". One of these
models should be selected and placed in a directory titled "base_models_(application)"

### Full Model 

```bash
# test TSN
python3.6 execute.py --help
```

## Pre-trained Models

All trained models and base models can be downloaded from here: [insert link]

## Analysis

```bash
# test backbone models
python3.6 analysis/backbone_analyze.py

# test general model architecture
python3.6 analysis/model_analyze.py
```