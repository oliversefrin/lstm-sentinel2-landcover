"""
This module contains several functions to be used
in a pipeline for hyperspectral image classification:

-------------------------------------------------
preprocessing_image_ms: scale hyperspectral image

simple_image_generator: create an image generator (needed to train a CNN)

time_series_image_generator: create an image generator that produces
            time sequences of images (needed to train a CNN+LSTM network)
"""

from glob import glob
from random import sample, choice
import numpy as np
from skimage.io import imread
from skimage.transform import rotate
from tensorflow.keras.utils import to_categorical


def preprocessing_image_ms(x):
    """"
    Scale Sentinel-2 data.

    Every channel is scaled individually to have a mean of 0 and a standard deviation of 1.
    Values are taken from the 3 timepoints 28-08-2016, 06-12-2016 and 24-04-2017.

    Args:
        x(np.array): Sentinel-2 data in arbitrary shape, except the channels are the last axis.

    Returns:
        x(np.array): Scaled Sentinel-2 data in the same shape as the input array.
    """
    # define mean and std values (taken from full multitemporal training set)
    mean = [1624.433, 1331.316, 1133.162, 1004.656, 1274.255,
            2089.642, 2409.098, 2368.672, 1001.349,   15.249,
            1292.258,  744.962, 2595.425
           ]

    std = [ 960.795, 1049.623,  995.824, 1202.114, 1217.579,
           1298.338, 1380.193, 1361.936,  544.678,    8.625,
            958.988,  667.875, 1438.869
          ]

    # loop over image channels
    for idx, mean_value in enumerate(mean):
        x[..., idx] -= mean_value
        x[..., idx] /= std[idx]
    return x


def simple_image_generator(
        path_to_data,
        dates,
        split,
        n_classes,
        batch_size=32,
        level='L1C',
        rotation_range=0,
        horizontal_flip=False,
        vertical_flip=False
):
    """
    Adapted image generator from
    https://github.com/jensleitloff/CNN-Sentinel/blob/master/py/image_functions.py.

    Instead of the whole image, now each pixel is labelled with a one-hot-vector.
    """
    if level == 'L1C':
        n_features = 13
    else:
        n_features = 17

    files = []
    for date in dates:
        files += glob(path_to_data+f'/{date}_data/{date}_{level}_{split}_tiles_(32x32)/*.tif')

    while True:
        # select batch_size number of samples without replacement
        batch_files = sample(files, batch_size)

        # array for images
        batch_X = []
        batch_Y = []
        # loop over images of the current batch
        for _, input_path in enumerate(batch_files):
            image = np.array(imread(input_path), dtype=float)
            # scale sentinel bands
            image[:, :, :n_features] = preprocessing_image_ms(image[:, :, :n_features])
            # process image
            if horizontal_flip:
                # randomly flip image up/down
                if choice([True, False]):
                    image = np.flipud(image)
            if vertical_flip:
                # randomly flip image left/right
                if choice([True, False]):
                    image = np.fliplr(image)
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range is not 0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                image = rotate(image, angle, mode='reflect',
                               order=1, preserve_range=True)

            # one hot encoding of labels
            # classes range from 1 to 5, but to_categorical counts from 0
            # therefore 0th index of last axis is omitted
            Y_one_hot = to_categorical(image[:, :, n_features], num_classes=n_classes+1)[:, :, 1:]
            # put all together
            batch_X += [image[:, :, :n_features]]
            batch_Y += [Y_one_hot]
        # convert lists to np.array
        X = np.array(batch_X)
        Y = np.array(batch_Y)

        yield(X, Y)



def time_series_image_gen(
        path_to_training_data,
        dates,
        n_classes,
        split='train',
        level='L1C',
        batch_size=32,
        rotation_range=0,
        horizontal_flip=False,
        vertical_flip=False
):
    """
    Adapted image generator from
    github.com/jensleitloff/CNN-Sentinel/blob/master/py/image_functions.py.

    Instead of the whole image, now each pixel is labelled
    with a one-hot-vector.
    The output arrays are now 4D: tiles are picked from different images
    that should be ordered chronologially in the 'dates' argument.

    Args:
        path_to_training_data (str):
            Indicates the directory that contains the subdirectories
            for different dates.

        dates (list):
            List of strings with the dates of the different pictures
            in format 'YYYYMMDD'.

        split (str):
            Indicates the subset split from which tiles are taken
            (train, valid or test).

        n_classes (int):
            Number of classes that are considered in the classification.

        level (str):
            Processing level, either L1C or L2A.

        batch_size (int):
            Size of the batch that the generator yields in one iteration.

        rotation_range (int):
            Rotation angle by which image is rotated at most (in degrees).

        horizontal_flip (bool):
            If True, allows chance of flipping the image horizontally.

        vertical_flip (bool):
            If True, allows chance of flipping the image vertically.

    Returns:
        X (np.array):
            Numpy array with shape
            (batch_size, timesteps, img_size, img_size, n_features)
            that contains a batch of sequences of sentinel-2 image tiles.

        Y_one_hot (np.array):
            Numpy array with shape
            (batch_size, img_size, img_size, n_classes)
            that contains a batch of one-hot-encoded labels corresponding to
            the data in X.
    """
    timesteps = len(dates)

    if level == 'L1C':
        n_features = 13
    else:
        n_features = 17

    # create path to the tiles and get filenames
    # directory structure expected to be
    # path_to_training_data/{date}_data/{split}_data
    # only take last 12 characters from each string
    # = '_row_col.tif'
    files = []
    for i, date in enumerate(dates):
        path_to_tiles = path_to_training_data+f'{date}_data/{date}_{level}_{split}_tiles_(32x32)'
        files.append([x[-12:] for x in glob(path_to_tiles+'/*.tif')])

    # make sure that tile exists in every satellite image
    # if not, drop it
    files_final = []
    for j in range(len(files[0])):
        exists_at_all_dates = True
        for i in range(1, timesteps):
            if files[0][j] not in files[i][:]:
                exists_at_all_dates = False
        if exists_at_all_dates:
            files_final.append(files[0][j])

    while True:
        X = np.empty((batch_size, len(dates), 32, 32, n_features))
        Y = np.empty((batch_size, 32, 32))
        Y_one_hot = np.empty((batch_size, 32, 32, n_classes))

        # draw samples
        batch = sample(files_final, batch_size)

        for i, date in enumerate(dates):
            # build filenames from endings
            batch_files = [f'{path_to_training_data}{date}_data/{date}_{level}_{split}_tiles_(32x32)/{date}_{level}_tile{file_ending}'
                           for file_ending in batch]

            # array for images
            batch_X = []
            # loop over images of the current batch
            for j, input_path in enumerate(batch_files):
                image = np.array(imread(input_path), dtype=float)
                # scale sentinel bands
                image[:, :, :13] = preprocessing_image_ms(image[:, :, :13])
                # put all together
                batch_X += [image[:, :, :13]]
                # ground truth is the same for every training date
                if i == 0:
                    Y[j, ...] = np.array(image[:, :, 13])
            X[:, i, ...] = np.array(batch_X)



        # image augmentation and one-hot-encoding
        # make sure to apply the same augmentation to every date
        # 1. outer loop: sample nr in batch
        # 2. random decision of image augmentation params
        # 3. inner loop: apply augmentation with params to all dates
        for s in range(batch_size):
            if horizontal_flip:
                # randomly flip image up/down
                if choice([True, False]):
                    for d in range(timesteps):
                        for f in range(n_features):
                            X[s, d, ..., f] = np.flipud(X[s, d, ..., f])
                    Y[s, ...] = np.flipud(Y[s, ...])
            if vertical_flip:
                # randomly flip image left/right
                if choice([True, False]):
                    for d in range(timesteps):
                        for f in range(n_features):
                            X[s, d, ..., f] = np.fliplr(X[s, d, ..., f])
                    Y[s, ...] = np.fliplr(Y[s, ...])
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range is not 0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                for d in range(timesteps):
                    for f in range(n_features):
                        X[s, d, ..., f] = rotate(X[s, d, ..., f],
                                                 angle,
                                                 mode='reflect',
                                                 order=1,
                                                 preserve_range=True
                                                 )
                Y[s, ...] = rotate(Y[s, ...],
                                   angle,
                                   mode='reflect',
                                   order=1,
                                   preserve_range=True
                                   )

            # one hot encoding of labels
            # classes range from 1 to n_classes, but to_categorical counts from 0
            # therefore 0th index of last axis is omitted

            Y_one_hot[s, ...] = to_categorical(Y[s, ...],
                                               num_classes=n_classes+1
                                               )[:, :, 1:]

        yield (X, Y_one_hot)



def random_sequence_gen(
        path_to_training_data,
        dates,
        timesteps,
        n_classes,
        split='train',
        level='L1C',
        batch_size=32,
        rotation_range=0,
        horizontal_flip=False,
        vertical_flip=False
):
    if level == 'L1C':
        n_features = 13
    else:
        n_features = 17

    while True:

        # select a number of dates
        # and order them
        # to get a sequence of length 'timesteps'
        sample_dates = [
            dates[i] for i in sorted(sample(range(len(dates)), timesteps))
        ]

        # create path to the tiles and get filenames
        # directory structure expected to be
        # path_to_training_data/{date}_data/{split}_data
        # only take last 12 characters from each string
        # = '_row_col.tif'
        files = []
        for i, date in enumerate(sample_dates):
            path_to_tiles = path_to_training_data+f'{date}_data/{date}_{level}_{split}_tiles_(32x32)'
            files.append([x[-12:] for x in glob(path_to_tiles+'/*.tif')])

        # make sure that tile exists in every satellite image
        # if not, drop it
        files_final = []
        for j in range(len(files[0])):
            exists_at_all_dates = True
            for i in range(1, timesteps):
                if files[0][j] not in files[i][:]:
                    exists_at_all_dates = False
            if exists_at_all_dates:
                files_final.append(files[0][j])

        X = np.empty((batch_size, timesteps, 32, 32, n_features))
        Y = np.empty((batch_size, 32, 32))
        Y_one_hot = np.empty((batch_size, 32, 32, n_classes))

        # draw samples
        batch = sample(files_final, batch_size)

        for i, date in enumerate(sample_dates):
            # build filenames from endings
            batch_files = [f'{path_to_training_data}{date}_data/{date}_{level}_{split}_tiles_(32x32)/{date}_{level}_tile{file_ending}'
                           for file_ending in batch]

            # array for images
            batch_X = []
            # loop over images of the current batch
            for j, input_path in enumerate(batch_files):
                image = np.array(imread(input_path), dtype=float)
                # scale sentinel bands
                image[:, :, :13] = preprocessing_image_ms(image[:, :, :13])
                # put all together
                batch_X += [image[:, :, :13]]
                # ground truth is the same for every training date
                if i == 0:
                    Y[j, ...] = np.array(image[:, :, 13])
            X[:, i, ...] = np.array(batch_X)



        # image augmentation and one-hot-encoding
        # make sure to apply the same augmentation to every date
        # 1. outer loop: sample nr in batch
        # 2. random decision of image augmentation params
        # 3. inner loop: apply augmentation with params to all dates
        for s in range(batch_size):
            if horizontal_flip:
                # randomly flip image up/down
                if choice([True, False]):
                    for d in range(timesteps):
                        for f in range(n_features):
                            X[s, d, ..., f] = np.flipud(X[s, d, ..., f])
                    Y[s, ...] = np.flipud(Y[s, ...])
            if vertical_flip:
                # randomly flip image left/right
                if choice([True, False]):
                    for d in range(timesteps):
                        for f in range(n_features):
                            X[s, d, ..., f] = np.fliplr(X[s, d, ..., f])
                    Y[s, ...] = np.fliplr(Y[s, ...])
            # rotate image by random angle between
            # -rotation_range <= angle < rotation_range
            if rotation_range is not 0:
                angle = np.random.uniform(low=-abs(rotation_range),
                                          high=abs(rotation_range))
                for d in range(timesteps):
                    for f in range(n_features):
                        X[s, d, ..., f] = rotate(X[s, d, ..., f],
                                                 angle,
                                                 mode='reflect',
                                                 order=1,
                                                 preserve_range=True
                                                 )
                Y[s, ...] = rotate(Y[s, ...],
                                   angle,
                                   mode='reflect',
                                   order=1,
                                   preserve_range=True
                                   )

            # one hot encoding of labels
            # classes range from 1 to n_classes, but to_categorical counts from 0
            # therefore 0th index of last axis is omitted

            Y_one_hot[s, ...] = to_categorical(Y[s, ...],
                                               num_classes=n_classes+1
                                               )[:, :, 1:]

        yield (X, Y_one_hot)
