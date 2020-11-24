""""
Use a trained LSTM model to classify on 2018 data.

Execute as
    python classify.py MODEL_NAME
with MODEL_NAME: filename of desired model (without .h5 ending) or ensemble of models
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys

import fire
import numpy as np
import rasterio
from tensorflow.keras.models import load_model

sys.path.append('../..')
import src.helpers.evaluation_tools as et


def permutation_classification(model_name):
    """

    :param model_name:
    :return:
    """
    # classification dates
    combinations_dict = {
        0: ['20180330', '20180419', '20180703', '20180807', '20180926', '20181205'],
        1: ['20180330', '20180529', '20180703', '20180807', '20180926', '20181205'],
        2: ['20180419', '20180529', '20180703', '20180807', '20180926', '20181205'],
        3: ['20180330', '20180419', '20180703', '20180807', '20181031', '20181205'],
        4: ['20180330', '20180529', '20180703', '20180807', '20181031', '20181205'],
        5: ['20180419', '20180529', '20180703', '20180807', '20181031', '20181205']
    }

    for i in range(6):
        classify(
            model_name,
            dates=combinations_dict[i],
            id=i,
        )


def classify(
        model_name,
        dates=['20180330', '20180419', '20180703', '20180807', '20180926', '20181205'],
        save_argmax=True,
        save_max=False,
        **kwargs
):
    """
    Get the classification and confidence map of a model for a given sequence of dates.

    :param save_max:
    :param save_argmax:
    :param model_name:
    :param dates:
    :return:
    """

    if 'id' in kwargs:
        id_str = '_' + str(kwargs['id'])
    else:
        id_str = ''

    if 'ensemble' in model_name:
        base_model, _, _ = model_name.partition('_ensemble')
        _, _, model_nums = model_name.rpartition('_')

        classification = np.zeros((2828, 3500, 8), dtype=np.float64)

        for num in list(model_nums):
            model = load_model(f'../../models/{base_model}_0{num}.h5')

            classification += et.get_classification(model, dates)

        classification /= len(model_nums)

    else:
        if 'model' in kwargs:
            model = kwargs['model']
        else:
            model = load_model(f'../../models/{model_name}.h5')

        classification = et.get_classification(model, dates)

    classification_map = np.zeros((2828, 3500), dtype=np.uint8)
    # argmax to get most probable class
    for i in range(classification_map.shape[0]):
        for j in range(classification_map.shape[1]):
            if np.any(classification[i, j, :]):
                classification_map[i, j] = np.argmax(classification[i, j, :]) + 1.

    # max to get value of highest probability
    confidence_map = np.max(classification, axis=-1).astype(np.float32)

    # save georeferenced
    path_to_output = f'../../data/processed/classified/{model_name}'
    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)
    sentinel_meta = rasterio.open('../../data/processed/ground_truth/gt.tif').profile

    # classification (all classes included)
    with rasterio.open(
            f'{path_to_output}/{sys.argv[1]}_classified{id_str}.tif',
            'w',
            driver=sentinel_meta['driver'],
            height=classification.shape[0],
            width=classification.shape[1],
            count=classification.shape[-1],
            dtype=classification.dtype,
            crs=sentinel_meta['crs'],
            transform=sentinel_meta['transform']
    ) as new_dataset:
        for i in range(8):
            new_dataset.write(classification[..., i], i+1)

    if save_argmax:
        # argmax (classification map)
        with rasterio.open(
                f'{path_to_output}/{sys.argv[1]}_class_2018_argmax{id_str}.tif',
                'w',
                driver=sentinel_meta['driver'],
                height=classification_map.shape[0],
                width=classification_map.shape[1],
                count=1,
                dtype=classification_map.dtype,
                crs=sentinel_meta['crs'],
                transform=sentinel_meta['transform']
        ) as new_dataset:
            new_dataset.write(classification_map, 1)

    if save_max:
        # max (confidence map)
        with rasterio.open(
                f'{path_to_output}/{sys.argv[1]}_class_2018_max{id_str}.tif',
                'w',
                driver=sentinel_meta['driver'],
                height=confidence_map.shape[0],
                width=confidence_map.shape[1],
                count=1,
                dtype=confidence_map.dtype,
                crs=sentinel_meta['crs'],
                transform=sentinel_meta['transform']
        ) as new_dataset:
            new_dataset.write(confidence_map, 1)


if __name__ == '__main__':
    fire.Fire(permutation_classification)
