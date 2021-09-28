# Temporal Feature LfD: D-ITR-L

## Overview

We release the PyTorch code of Deep Interval Temporal Relationship Learner (D-ITR-L)

## Contents

- [Prerequisites](#prerequisites)
- [Downloads](#downloads)
  - [Datasets](#dataset)
  - [Pretrained Models](#dataset)
- [Implementation](#code)
  - [Training Backbone Models](#training-backbone-models)
  - [Training Baseline Temporal Inference Architectures](#training-baseline-temporal-inference-architectures)
  - [Training D-ITR-L](#Training-D-ITR-L)
  - [Evaluation](#evaluation)

## Prerequisites

The code is built with following libraries:

- Python 3.6 or later
- [PyTorch](https://pytorch.org/) 1.7 or later
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [openCV](hhttps://pypi.org/project/opencv-python/) 4.5 or later

## Downloads

### Dataset

This work has been designed to interface with three datasets, one policy learning architecture (Block Stacking)
and two activity recognition datasets (Furniture Construction and Recipe Following). We provided the frame
by frame versions of the datasets here the source videos can be downloaded at the links in the following table.

| Dataset      | Source |  Frames |
| ----------- | ----------- | ----------- |
| Block Stacking (Fixed Timing)    | link       | link       |
| Block Stacking (Variable Timing)     | link       | link       |
| IKEA Furniture Assembly   | link        | link        |
| Crepe Recipe Following (Sub-Actions)   | --        | link        |
| Crepe Recipe Following (Full-Recipes)   | link        | link        |

### Pre-trained Models

For re-producibility we provide several trained models. When training the backbone models we generally leveraged
pre-trained features from architectures exposed to the ImageNet dataset. PyTorch provides internal models for the VGG-16 
and Wide ResNet architectures but the other two of the backbone models investigated in this work (Temporal Shift Module 
and I3D) leverage external models. We provide links to where those datasets can be downloaded from.


| Backbone Model      | Source |  
| ----------- | ----------- | 
| Temporal Shift Module     | link       | 
| I3D   | link        | 

We provide zip files containing the trained spatial and temporal features of models investigated in this work.

| Dataset      | Trained Models  |
| ----------- | ----------- | 
| Block Stacking (Fixed Timing)    | link       | 
| Block Stacking (Variable Timing)     | link       |
| IKEA Furniture Assembly   | link        | 
| Crepe Recipe Following (Sub-Actions)   | link        | 
| Crepe Recipe Following (Full-Recipes)   | link        |

## Implementation

The implementation of this model is distributed into several sections: 
1. Training of the backbone model to identify significant spatial features present in the dataset
2. Training of the temporal inference architectures to learn temporal representations from the identified spatial features.
3. Evaluation of the trained models

All of the executables can be run with the `--help` flag to pull up a list of legal parameters.

### Training Backbone Models

The backbone models can be trained by running the `execute_backbone.py` file. 

```bash
# Running the program
python3.6 execute_backbone.py --help

# Example Execution
python3.6 execute_backbone.py 
```

Trained models will be placed in folders of the type "saved\_model\_(application)\_(bottleneck_size)". One of these
models should be selected and placed in a directory titled "base\_models\_(application)"

### Training Baseline Temporal Inference Architectures

```bash
# test TSN
python3.6 execute.py --help
```

### Training D-ITR-L

```bash
# test TSN
python3.6 execute.py --help
```


### Evaluation

```bash
# test backbone models
python3.6 analysis/backbone_analyze.py

# test general model architecture
python3.6 analysis/model_analyze.py
```