![GitHub](https://img.shields.io/github/license/oliversefrin/lstm-sentinel2-landcover)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4289079.svg)](https://doi.org/10.5281/zenodo.4289079)

# Code for _Deep Learning for Land Cover Change Detection_

This repo contains the code for the pre-processing, model training, and
classification evaluation described in the article
[_Deep Learning for Land Cover Change Detection_](https://doi.org/10.3390/rs13010078).
Using a land cover vector ground truth (GT) and freely available Sentinel-2
imagery, we train different neural network classifiers. We employ a U-Net based
fully convolutional network (FCN) as a base model and present a method of adding
a convolutional LSTM cell to make predictions on sequences of satellite images.
This LSTM approach shows significant improvements on seasonally varying classes
such as _grassland_ or _farmland_. Other contributions include an approach to
mask inconsistent water shorelines based on the NDWI and a method to prevent
the exclusion of incomplete image tiles with a specially weighted loss function.

**License:** [3-Clause BSD license](LICENSE)

**Authors:**

* [Oliver Sefrin](https://github.com/oliversefrin)
* [Felix M. Riese](https://github.com/felixriese)
* [Sina Keller](https://github.com/sinakeller)

**Citation:** See [citation](#Citation) and [bibliography.bib](bibliography.bib).


## Content
1. [Python Environment Setup](#Python-Environment-Setup)
1. [Data Placement](#Data-Placement)
1. [Pre-Processing](#Pre-Processing)
1. [Model Training](#Model-Training)
1. [Evaluation Metrics](#Evaluation-Metrics)
1. [Citation](#Citation)


## Python Environment Setup
To set up the Python environment, execute
1. `conda create -n n_env python=3.7 pip`
1. `conda activate n_env`
1. `pip install -r requirements.txt`
1. `conda install --yes --file requirements_conda.txt` (we recommend installing GDAL and Rasterio via _conda_)


## Data Placement
* Sentinel-2 images: `data/raw/sentinel/`
* Ground Truth (GT) shapefile: `data/raw/ground_truth/gt_shapefile/`
* shapefile of area of interest (AOI) = shape of GT as shapefile: `data/raw/ground_truth/overallshape/`


## Pre-Processing
All pre-processing functionality is in `src/data_processing/`. The main features are:
* clipping of Sentinel-2 image to the AOI (`01_clip_sentinel.sh`)
* rasterization of the GT shapefile (`02_rasterize_shapefile.sh`)
* creation of training data by merging image and GT (`04_merge_tifs.sh`)
* creation of prediction data by merging image and AOI GeoTIFF file (`04_merge_tifs.sh`)
* Tile Splitting to accomodate FCN structure

## Model Training
Example models are in `src/models/`, the data generators used for training are in `src/helpers/image_functions.py`.

+ `baseline_cnn.py` is a simple FCN that trains on single images at a time (i.e. no sequences). It can be used as a pretrained base model for the LSTM models, as shown in e.g. `lstm_fixed_seq.py`.
+ `lstm_fixed_seq.py` is a FCN+LSTM that trains with sequences of images, however the images that build the sequence are fixed (``.
+ `lstm_random_seq.py` is also a FCN+LSTM that randomly builds a new sequence of images for each training batch.

Trained models are saved in `models/`.
Tensorboard logs are saved in separate subdirectories of `reports/logs/`; this way they can be called via `tensorboard --logdir=reports/logs` (when in main directory).

## Evaluation Metrics
To get the accuracy scores and the confusion matrix of a model, execute `python metrics.py MODEL_NAME` with *MODEL_NAME* being the filename of a model saved in the models directory (without .h5 file ending).

For easy use, model names should follow the name convention:
```
If the model uses temporal sequences as input data, the name must contain 'lstm'.
If it uses non-sequence input, the name should not contain 'lstm'.
```

An easy way to evaluate a model right after training is to add the function call `get_metrics_and_cm($model_name)` to the end of the training script. For this to work, the function should be imported as follows:
```
import sys
sys.path.append('../visualization')
from metrics import get_metrics_and_cm
```

## Citation

**Code:**

```tex
@misc{sefrin2020code,
    author = {Sefrin, Oliver and Riese, Felix~M. and Keller, Sina},
    title = {{Code for Deep Learning for Land Cover Change Detection}},
    year = {2020},
    publisher = {Zenodo},
    doi = {10.5281/zenodo.4289079},
}
```

**Paper:**

Sefrin, Oliver; Riese, Felix M.; Keller, Sina. 2021. "Deep Learning for Land
Cover Change Detection" Remote Sens. 13, no. 1: 78.

```tex
@article{sefrin2021deep,
    author = {Sefrin, Oliver and Riese, Felix~M. and Keller, Sina},
    title = {{Deep Learning for Land Cover Change Detection}},
    journal = {Remote Sensing},
    year = {2021},
    volume = {13},
    number = {1},
    article-number = {78},
    DOI = {10.3390/rs13010078 },
    publisher={Multidisciplinary Digital Publishing Institute},
}
```
