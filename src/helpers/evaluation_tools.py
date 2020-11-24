import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score
sys.path.append('../..')
from src.helpers.image_functions import preprocessing_image_ms


def get_seq_val_or_test_data(dates, split='valid', level='L1C', cutoff=0):
    """
    Load the validation data as sequences
    in the format suitable for the Neural Net.
    """
    if split == 'valid':
        print('Loading and processing validation data...')
    else:
        print('Loading and processing test data...')

    timesteps = len(dates)

    # processing level of Sentinel-2 images
    if level == 'L1C':
        n_features = 13
    else:
        n_features = 17

    # define path to training data
    path_to_training_data = '../../data/processed/training_data/'

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

    n_tiles = len(files_final)

    validation_data = np.empty((n_tiles, timesteps, 32-2*cutoff, 32-2*cutoff, n_features+1))

    for i, file in enumerate(files_final):
        for j, date in enumerate(dates):
            path_to_tile = f'../../data/processed/training_data/{date}_data/{date}_{level}_{split}_tiles_(32x32)/{date}_{level}_tile'+file
            validation_data[i, j, ...] = imread(path_to_tile)[cutoff:32-cutoff, cutoff:32-cutoff, :]
            validation_data[i, j, ..., :n_features] = preprocessing_image_ms(validation_data[i, j, ..., :n_features])

    X_val = validation_data[..., :n_features]
    Y_val = validation_data[:, 0, :, :, n_features]

    return X_val, Y_val


def get_val_or_test_data(dates, split='valid', level='L1C', cutoff=0):
    """
    Load validation data for one or several dates in the suitable form
    for the baseline CNN.
    """
    print(f'Loading and processing "{split}" subset data...')

    # processing level of Sentinel-2 images
    if level == 'L1C':
        n_features = 13
    else:
        n_features = 17

    # define path to training data
    path_to_training_data = '../../data/processed/training_data/'

    files = []
    for i, date in enumerate(dates):
        path_to_tiles = path_to_training_data+f'{date}_data/{date}_{level}_{split}_tiles_(32x32)'
        for x in glob(path_to_tiles + '/*.tif'):
            files.append(x)

    n_tiles = len(files)

    data = np.empty((n_tiles, 32-2*cutoff, 32-2*cutoff, n_features+1))

    for i, file in enumerate(files):
        data[i, ...] = imread(file)[cutoff:32-cutoff, cutoff:32-cutoff, :]
        data[i, ..., :n_features] = preprocessing_image_ms(data[i, ..., :n_features])

    X_val = data[..., :n_features]
    Y_val = data[..., n_features]

    return X_val, Y_val


def print_metrics(y_true, y_pred, model_name):
    """
    Print some metrics to evaluate the classification
    and write to a log file.
    """
    if not os.path.exists(f'../../reports/logs_and_plots/{model_name}'):
        os.makedirs(f'../../reports/logs_and_plots/{model_name}')

    # scores
    with open(f'../../reports/logs_and_plots/{model_name}/{model_name}_log.txt', 'a') as file:
        file.write(f'Accuracy score:               {accuracy_score(y_true, y_pred):0.5f}\n')
        file.write(f'F1 score (weighted):          {f1_score(y_true, y_pred, average="weighted"):0.5f}\n')
        file.write('F1 score (class-wise):         ')
        for x in f1_score(y_true, y_pred, average=None):
            file.write(f'{x:0.5f} ')
        file.write('\n')
        file.write(f'Precision score (weighted):   {precision_score(y_true, y_pred, average="weighted"):0.5f}\n')
        file.write('Precision score (class-wise):  ')
        for x in precision_score(y_true, y_pred, average=None):
            file.write(f'{x:0.5f} ')
        file.write('\n')
        file.write(f'Recall score (weighted):      {recall_score(y_true, y_pred, average="weighted"):0.5f}\n')
        file.write('Recall score (class-wise):     ')
        for x in recall_score(y_true, y_pred, average=None):
            file.write(f'{x:0.5f} ')
        file.write('\n')
        file.write(f'Jaccard score (weighted):     {jaccard_score(y_true, y_pred, average="weighted"):0.5f}\n')
        file.write('Jaccard score (class-wise):    ')
        for x in jaccard_score(y_true, y_pred, average=None):
            file.write(f'{x:0.5f} ')
        file.write('\n')
        file.write(f'Cohen kappa score:            {cohen_kappa_score(y_true, y_pred):0.5f}\n')

    # print to screen
    print(f'Accuracy score:               {accuracy_score(y_true, y_pred):0.5f}')
    print(f'F1 score (weighted):          {f1_score(y_true, y_pred, average="weighted"):0.5f}')
    print('F1 score (class-wise):        ', *(f'{x:0.5f}' for x in f1_score(y_true, y_pred, average=None)))
    print(f'Precision score (weighted):   {precision_score(y_true, y_pred, average="weighted"):0.5f}')
    print('Precision score (class-wise): ', *(f'{x:0.5f}' for x in precision_score(y_true, y_pred, average=None)))
    print(f'Recall score (weighted):      {recall_score(y_true, y_pred, average="weighted"):0.5f}')
    print('Recall score (class-wise):    ', *(f'{x:0.5f}' for x in recall_score(y_true, y_pred, average=None)))
    print(f'Jaccard score (weighted):     {jaccard_score(y_true, y_pred, average="weighted"):0.5f}')
    print('Jaccard score (class-wise):   ', *(f'{x:0.5f}' for x in jaccard_score(y_true, y_pred, average=None)))
    print(f'Cohen kappa score:            {cohen_kappa_score(y_true, y_pred):0.5f}')


def plot_confusion_matrix(
        y_true,
        y_pred,
        classes,
        test_name,
        normalize=False,
        set_title=False,
        save_fig=False,
        cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if set_title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # and save it to log file
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        with open(f'../../reports/logs_and_plots/{test_name}/{test_name}_log.txt', 'ab') as f:
            f.write(b'\nNormalized confusion matrix\n')
            np.savetxt(f, cm, fmt='%.3f')
    else:
        print('Confusion matrix, without normalization')
        with open(f'../../reports/logs_and_plots/{test_name}/{test_name}_log.txt', 'ab') as f:
            f.write(b'\nConfusion matrix, without normalization\n')
            np.savetxt(f, cm, fmt='%7u')

    print(cm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if normalize:
        im.set_clim(0., 1.)     # fixes missing '1.0' tick at top of colorbar
    cb = ax.figure.colorbar(im, ax=ax)
    if normalize:
        cb.set_ticks(np.arange(0., 1.2, 0.2))
        cb.set_ticklabels([f'{i/5:.1f}' for i in range(6)])
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title if set_title else None,
           ylabel='True label',
           xlabel='Predicted label')
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if np.round(cm[i, j], 2) > 0.:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, '–',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if save_fig:
        if normalize:
            plt.savefig(f'../../reports/logs_and_plots/{test_name}/{test_name}_cm_normal.pdf')
        else:
            plt.savefig(f'../../reports/logs_and_plots/{test_name}/{test_name}_cm_non_normal.pdf')
    return fig, ax


def get_classification(
        model,
        dates,
        level='L1C'
):
    """
    Classify new data and store it as a numpy array.

    Returns in shape (height, width, n_classes), i.e. before applying
    np.argmax.
    """
    timesteps = len(dates)

    # read full area of interest
    area = (imread('../../data/processed/ground_truth/area.tif') +
            imread('../../data/processed/ground_truth/czech_area.tif')).astype(bool)

    # processing level of Sentinel-2 images
    if level == 'L1C':
        n_features = 13
    else:
        n_features = 17

    # number of classes
    n_classes = 8

    # define path to new data
    path_to_new_data = '../../data/processed/classification_data/'

    files = []
    for i, date in enumerate(dates):
        path_to_tiles = path_to_new_data+f'{date}_data/{date}_{level}_all_tiles_(32x32)'
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
    n_tiles = len(files_final)

    # make pseudo-batch of one tile per batch
    # to fit input shape of the neural net
    classification_data = np.empty((1, timesteps, 32, 32, n_features))

    # array where classified data is stored
    classification_map = np.zeros((2828, 3500, n_classes), dtype=np.float)

    for file in tqdm(files_final, desc='Predict tiles'):
        for j, date in enumerate(dates):
            path_to_tile = f'../../data/processed/classification_data/{date}_data/{date}_{level}_all_tiles_(32x32)/{date}_{level}_tile'+file
            classification_data[0, j, ...] = imread(path_to_tile)[..., :n_features]
            classification_data[0, j, ...] = preprocessing_image_ms(classification_data[0, j, ...])

        row_nr, col_nr = int(file[1:4]), int(file[5:8])

        pred_tile = model.predict(classification_data)

        # note that we use overlapping tiles
        # therefore only the 16x16 center is written to the classification map
        classification_map[
            row_nr*16 + 8: (row_nr*16) + 24,
            col_nr*16 + 8: (col_nr*16) + 24,
            :
        ] = pred_tile[0, 8:24, 8:24, :]

        # set pixel outside AOI to 0 again
        classification_map[~area] = 0

    return classification_map
