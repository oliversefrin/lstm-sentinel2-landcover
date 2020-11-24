import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys

import fire
import numpy as np
from tensorflow.keras.models import load_model

sys.path.append('../..')
import src.helpers.evaluation_tools as et
from src.helpers.dicts import inv_category_dict


def get_metrics_and_cm(model_name, split='test', **kwargs):
    """
    Loads a trained model and computes its metrics and the confusion matrix.

    :param model_name: Name of a saved model (without .h5 ending)
    :param split: whether valid or test subset is used for metrics calculation. Default: test
    :keyword model: tf.keras.Model instance
    :return:
    """
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        model = load_model(f'../../models/{model_name}.h5')

    # dates used to build validation data
    dates = [
        '20160403',
        '20160522',
        '20160828',
        '20160929',
        '20161118',
        '20161206'
    ]

    # if model is a LSTM model:
    #   validation data needs to be a sequence
    # else (simple CNN):
    #   validation data is just a stack of tiles from all the dates
    if 'lstm' in model_name:
        x_val, y_val = et.get_seq_val_or_test_data(dates, split=split)
    else:
        x_val, y_val = et.get_val_or_test_data(dates, split=split)

    # predict and shape
    y_pred = model.predict(x_val)
    y_pred = np.argmax(y_pred, axis=-1) + 1

    y_pred_flat = y_pred.flatten()
    y_val_flat = y_val.flatten()

    # make sure target directory exists
    if not os.path.exists(f'../../reports/logs_and_plots/{model_name}/'):
        os.makedirs(f'../../reports/logs_and_plots/{model_name}/')

    class_names = [inv_category_dict[i] for i in np.arange(1, 9)]

    # save plot of cm
    et.plot_confusion_matrix(
        y_val_flat,
        y_pred_flat,
        classes=class_names,
        test_name=model_name,
        save_fig=True
    )
    # save plot of normalized cm
    et.plot_confusion_matrix(
        y_val_flat,
        y_pred_flat,
        classes=class_names,
        test_name=model_name,
        normalize=True,
        save_fig=True
    )

    # empty line
    print("")

    # before calculating metrics,
    # delete all pixel of 'inclassifiable' category
    mask_array = np.ma.masked_equal(y_val_flat, 8.)
    y_pred_flat = y_pred_flat[~mask_array.mask]
    y_val_flat = y_val_flat[~mask_array.mask]

    # call print_metrics script
    et.print_metrics(y_val_flat, y_pred_flat, model_name)

    return None


if __name__ == '__main__':
    fire.Fire(get_metrics_and_cm)
