import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys

from fire import Fire
from glob import glob
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append('../..')
import src.helpers.evaluation_tools as et
from src.helpers.dicts import inv_category_dict


def get_ensemble_metrics(
        model_name: str,
        split: str = 'test',
        test_nums: str = None
):
    """
    Calculate metrics on an ensemble prediction of a model type.

    :param model_name: Namestring of trained model (without .h5 ending).
    :param split: Either test or valid, selection of subset used for evaluation.
    :param test_nums: Optionally indicate the test run numbers the ensemble should be build from,
        eg. '134' for tests 01, 03, 04.
    :return:
    """

    dates = [
        '20160403',
        '20160522',
        '20160828',
        '20160929',
        '20161118',
        '20161206'
    ]
    if test_nums is not None:
        tests_list = list(str(test_nums))
        tests = [f'../../models/{model_name}_0{num}.h5' for num in tests_list]
    else:
        tests = glob(f'../../models/{model_name}*')
        test_nums = ''.join(str(num) for num in range(1, len(tests)+1))

    test_name = f'{model_name}_ensemble_of_tests_{test_nums}'
    n_tests = len(tests)

    if not os.path.exists(f'../../reports/logs_and_plots/{test_name}/'):
        os.makedirs(f'../../reports/logs_and_plots/{test_name}/')

    if 'lstm' in model_name:
        x_test, y_test = et.get_seq_val_or_test_data(dates, split=split)
    else:
        x_test, y_test = et.get_val_or_test_data(dates, split=split)

    y_pred = np.zeros(y_test.shape + (8,))

    print(f'Loading {n_tests} models and predicting on each...')
    # for ensembles of more than 4, the GPU runs out of memory
    # therefore the CPU is used here
    with tf.device('/CPU:0'):
        for i, test in tqdm(enumerate(tests)):
            model = load_model(test)

            y_pred += model.predict(x_test)

        y_pred = y_pred / n_tests

    y_pred = np.argmax(y_pred, axis=-1) + 1

    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test.flatten()

    class_names = [inv_category_dict[i] for i in np.arange(1, 9)]

    et.plot_confusion_matrix(
        y_test_flat,
        y_pred_flat,
        classes=class_names,
        test_name=test_name,
        save_fig=True
    )

    et.plot_confusion_matrix(
        y_test_flat,
        y_pred_flat,
        classes=class_names,
        normalize=True,
        test_name=test_name,
        save_fig=True
    )

    # empty line
    print("")

    # before calculating metrics,
    # delete all pixel of 'inclassifiable' category
    mask_array = np.ma.masked_equal(y_test_flat, 8.)
    y_pred_flat = y_pred_flat[~mask_array.mask]
    y_test_flat = y_test_flat[~mask_array.mask]

    # get metrics
    et.print_metrics(y_test_flat, y_pred_flat, test_name)

    return None


if __name__ == '__main__':
    Fire(get_ensemble_metrics)
