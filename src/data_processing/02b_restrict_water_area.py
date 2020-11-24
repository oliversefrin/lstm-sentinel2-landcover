"""
This script restricts the water areas in the ground truth.

The NDWI (normalized difference water index) of all training images is
calculated.
Water in the Ground Truth is only kept as such if it appears as water in the
NDWI of every training image (i.e. if the NDWI is above the set threshold),
else the pixel's label is changed to 'nicht klassifizierbar'.
"""

from glob import glob
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imsave

# variables
NDWI_THRESHOLDS = np.arange(-0.6, 0.8, 0.2)

# dates of all training images
dates = [
    '20151231',
    '20160403',
    '20160522',
    '20160828',
    '20160929',
    '20161118',
    '20161206',
    '20170328',
    '20170424',
    '20170527'
]

# load ground truth
path_to_tif = '../../data/processed/ground_truth/ATKIS_mod.tif'
gt = imread(path_to_tif)

for NDWI_THRESHOLD in NDWI_THRESHOLDS:
    # mask water in gt
    water_gt = gt.copy()
    water_gt[water_gt != 7.0] = 0
    water_gt[water_gt == 7.0] = 1

    # loop over all dates
    # element-wise OR between water_gt and ndwi of image
    print('Calculating NDWI of every Sentinel 2 image')
    print('and comparing to Ground Truth...')
    for date in tqdm(dates):
        # calculate NDWI (normalized difference water index) for training images
        #
        # formula: ndwi = (Green - NIR) / (Green + NIR)
        # with Green: band 3
        # and NIR: band 8
        #
        # remember: ordering of bands is 1-12, then 8a
        path_to_sentinel = f'../../data/raw/sentinel/level_L1C/'
        path_to_sentinel = glob(path_to_sentinel + f'*{date}*.SAFE')[0]
        sentinel_data = imread(path_to_sentinel + f'/{date}_L1C_all_bands.tif')

        ndwi = np.zeros((sentinel_data.shape[:2]), dtype=np.float64)

        # care: original data type is uint
        green = sentinel_data[..., 2].astype(int)
        nir = sentinel_data[..., 7].astype(int)

        for i in range(sentinel_data.shape[0]):
            for j in range(sentinel_data.shape[1]):
                if (green[i, j] + nir[i, j]) != 0:
                    ndwi[i, j] = (green[i, j] - nir[i, j]) / (green[i, j] + nir[i, j])

        ndwi[ndwi >= NDWI_THRESHOLD] = 1
        ndwi[ndwi < NDWI_THRESHOLD] = 0

        water_gt = np.multiply(water_gt, ndwi)

    # set water pixel that don't appear as water in every image to garbage class 8
    gt[np.logical_and(gt == 7.0, water_gt == 0)] = 8.0

    # save modified ground truth
    imsave('../../data/processed/ground_truth/gt_water_masked_' + f'{NDWI_THRESHOLD:3.1}' + '.tif', gt)

    print('done.')
    print('Modified Ground Truth is saved as gt_water_masked.tif')
