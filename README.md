# Temporal Feature Learning from Demonstration: Deep Interval Temporal Relationship Learner (D-ITR-L)

Deep Interval Temporal Relationship Learner is an architecture for identifying and learning 
from temporal features as they are expressed in the video of human-led sequential tasks.

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
    - [Classification](#classification)
    - [Policy Learning](#policy-learning)

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

For reproducibility we provide several trained models. When training the backbone models we generally leveraged
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

All executables can be run with the `--help` flag to pull up a list of legal parameters.

### Prerequisities

You will likely need to update the directory paths listed on lines 18-20 of `parameter_parser.py` to point
to the directory where your datasets are located and the pretrained source files for fine-tuning the I3D and TSM models.

### Training Backbone Models

The backbone models can be trained by running the `execute_backbone.py` file with a specific backbone model identifier and application name. 
The available options are listed when run with the `--help` command. Included in the following code
is an example application for running training the Temporal Shift Module (tsm) on the IKEA furniture construction
dataset (ikea). The code will perform a grid search generating models that use the provided backbone model and dataset over different bottleneck sizes
Models will be generated at the bottleneck sizes of 8, 16, 32, and 64. Which can later be investigated and the best 
model can be used when conducting temporal inference. In the example I use the `--repeat` flag to fine-tune the model several times to sample
different architectures by leveraging deep learnings inherent stochasticity.

```bash
# Running the program
python3.6 execute_backbone.py <backbone_model_id> --application <application_name> --repeat <number_of_repitions>

# Example Execution
python3.6 execute_backbone.py tsm --application ikea --repeat 3
```

Trained models will be placed in folders of the type `saved_models_<bottleneck_size>\c_backbone_<backbone_model_id>_<repetition_identifier>`.
An example of the directories generated is presented below.

```bash
# example directory structure
saved_models_8/c_tsm_backbone_0
saved_models_8/c_tsm_backbone_1
saved_models_8/c_tsm_backbone_2
saved_models_16/c_tsm_backbone_0
...
saved_models_64/c_tsm_backbone_2
```

After training the models they can be evaluated using the following code. The application type is either
'c' for a classification task or 'pl' for policy learning. Training the backbone model should always 
be done as a classification task.

```bash
# Running the program
python3.6 analysis/model_analyze.py <application_type> <model_directory>

# Example Execution
python3.6 analysis/model_analyze.py c saved_models_8/c_vgg_backbone_0
```

After a model has been selected to perform spatial feature extraction for the task particular application it should be moved 
to a new directory. Create a directory titled `base_models_<application_name>`. Given our example we
would move setup our directories as follows: `base_models_ikea/c_vgg_backbone_0`. This 
directory name should be updated in the `parameter_parser.py` file. Line 107 captures the 
TSM directory name used for the 'ikea' application. Both it and the bottleneck size value should be updated
appropriately.

### Training the Inference Models

Once the backbone model has been established it is time to train the inference models. In our work
we investigated four approaches:
 - no temporal inference using a linear model (linear). 
 - a recurrent neural network: the long short-term memory cell (lstm)
 - a convolution over time base approach using Temporal Convolutional Network (tcn)
 - Deep Interval Temporal Relationship Network (ditrl)

#### Classification
Training of these models is accomplished through the following command:

```bash
# Running the program
python3.6 execute.py <application_type> <backbone_model_name> <inference_approach> --application <application_name>

# Example Execution
python3.6 execute.py c tsm ditrl --application ikea
```

This code uses the trained features of the fixed backbone model to identify feature presence in the input video. This information
is then passed to one of the temporal inference approaches. When conducting
inference using the temporal model the architecture will generate intermediary files (IADs) in the directory where the 
dataset is located in order to expedite learning. If using D-ITR-L for inference then graph files will be saved in the same location.
Be forewarned that these files can be quite large. 

The trained models are placed in a directory titled: 
`saved_models_<application>/<application_type>_<inference_approach>_<run_id>`. This file can be interrogated using the same model_analysis 
code as before.

#### Policy Learning

The execution code for policy learning is similar to that used for classification with the exception being the use of the
'pl' application_type.

```bash
# Example Execution
python3.6 execute.py pl tsm ditrl --application block_construction_timed
```