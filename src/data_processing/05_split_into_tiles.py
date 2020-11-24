"""
This script splits a Geotiff file into small tiles suited for training with a CNN.

Execute script as
    python 05_split_into_tiles.py DATE LEVEL USE
with
    DATE: date of sentinel-2 image in YYYYMMDD format
    LEVEL: processing level of sentinel-2 image, either L1C or L2A
    USE: either TRAIN for training of CNN or NEW for prediction
"""

# packages
import os
import sys
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.io import imsave
sys.path.append('../..')
import src.data_processing.subset_split as subsets


## set variables
# square tiles of shape (tile_size, tile_size)
tile_size = 32

## arguments defined upon calling python script
# date of sentinel-2 image
date = sys.argv[1]
# sentinel-2 processing level
level = sys.argv[2]
if level not in ['L1C', 'L2A']:
    sys.exit('Wrong input passed to argument "LEVEL".')

# use
if sys.argv[3] == 'TRAIN':
    is_training_data = True
    tile_separation = 1
elif sys.argv[3] == 'NEW':
    is_training_data = False
    tile_separation = 0
else:
    sys.exit('Wrong input passed to argument "USE".')


# construct paths
filename = f'{date}_{level}_merged.tif'
dirname = f'{date}_{level}_all_tiles_({tile_size}x{tile_size})'

if is_training_data:
    path = f'../../data/processed/training_data/{date}_data/'
else:
    path = f'../../data/processed/classification_data/{date}_data/'

path_to_all_tiles = os.path.join(path, dirname)

if not os.path.isdir(path_to_all_tiles):
    os.makedirs(path_to_all_tiles)


# split into tiles

print('\nsplitting image into tiles...', end='')
with rasterio.open(path+filename) as tif:
    height = tif.profile['height']
    width = tif.profile['width']

    if is_training_data:
        # expected: 85 rows, 106 cols
        nr_tile_rows = height // (tile_size + tile_separation)
        nr_tile_cols = width // (tile_size + tile_separation)

        for i in range(nr_tile_rows):
            for j in range(nr_tile_cols):
                # get a tile
                # Window syntax:
                # Window(col_off, row_off, width, height)
                tile = tif.read(window=Window(j*(tile_size+tile_separation),
                                              i*(tile_size+tile_separation),
                                              tile_size,
                                              tile_size)
                                )
                # get from shape=(channels, height, width)
                # to (height, width, channels)
                tile = np.swapaxes(tile, 0, 2)
                tile = np.swapaxes(tile, 0, 1)

                # for training data:
                # check if tile lies entirely in labelled area
                # --> no zeroes in ground truth (last band)
                # check if tile lies entirely in sentinel-2 area
                categories = tile[:, :, -1]
                if np.all(categories):
                    if np.all(tile[:, :, 0]):
                        imsave(
                            os.path.join(
                                path,
                                dirname,
                                f'{date}_{level}_tile_{i:03d}_{j:03d}.tif'
                            ),
                            tile
                        )

    # classification data:
    # make tile every half tile width
    # in order to keep center part of tile only for classification
    else:
        overlap = tile_size // 2
        nr_tile_rows = height // overlap
        nr_tile_cols = width // overlap

        for i in range(nr_tile_rows):
            for j in range(nr_tile_cols):
                # get a tile
                # Window syntax:
                # Window(col_off, row_off, width, height)
                tile = tif.read(window=Window(j*overlap,
                                              i*overlap,
                                              tile_size,
                                              tile_size)
                                )
                # get from shape=(channels, height, width) to (height, width, channels)
                tile = np.swapaxes(tile, 0, 2)
                tile = np.swapaxes(tile, 0, 1)

                # check if center 16x16 intersects area of interest
                if np.any(np.logical_or(tile[8:24, 8:24, -2],
                                        tile[8:24, 8:24, -1])):
                    # check if tile on sentinel image
                    if np.all(tile[:, :, 0]):
                        imsave(
                            os.path.join(
                                path,
                                dirname,
                                f'{date}_{level}_tile_{i:03d}_{j:03d}.tif'
                            ),
                            tile
                        )

print('done.')


if is_training_data:
    print('shuffling tiles into subset splits...', end='')
    subsets.make_subset_split(date, level)
    print('done.')

    print(f'deleting directory "{dirname}"...', end='')
    shutil.rmtree(path_to_all_tiles)
    print('done.')

# os.remove(os.path.join(path, filename))
