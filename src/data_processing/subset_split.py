"""
The function in this script creates a train, test and validation subset split.

The script can also be executed by itself with
    python subset_split.py DATE LEVEL
with
    DATE:   date of sentinel-2 image in YYYYMMDD format
    LEVEL:  processing level of sentinel-2 image, either L1C or L2A
"""

import os
import sys
import random
import shutil


def make_subset_split(date, level, valid_frac=0.2, test_frac=0.2, tile_size=32):
    """
    This function splits all tiles into a train, test and validaton split.

    Args:
        date (str):         date of sentinel-2 image in YYYYMMDD format
        level (str):        processing level of sentinel-2 image, either L1C or L2A
        valid_frac (float): fraction of tiles to put in validation subset
        test_frac (float):  fraction of tiles to put in test subset
        tile_size (int):    tilesize of tiles
    """

    path = f'../../data/processed/training_data/{date}_data/'
    path_to_all_tiles = os.path.join(path,
                                     f'{date}_{level}_all_tiles_({tile_size}x{tile_size})')
    path_to_train_tiles = os.path.join(path,
                                       f'{date}_{level}_train_tiles_({tile_size}x{tile_size})')
    path_to_validation_tiles = os.path.join(path,
                                            f'{date}_{level}_valid_tiles_({tile_size}x{tile_size})')
    path_to_test_tiles = os.path.join(path,
                                      f'{date}_{level}_test_tiles_({tile_size}x{tile_size})')

    paths = [path_to_all_tiles, path_to_train_tiles, path_to_validation_tiles, path_to_test_tiles]
    for path_to_dir in paths:
        if not os.path.isdir(path_to_dir):
            os.mkdir(path_to_dir)

    all_tile_endings = [line.rstrip('\n') for line in open('../../data/processed/all_tile_endings.txt')]

    files = [path_to_all_tiles+f'/{date}_{level}_tile'+ending for ending in all_tile_endings]

    split_index_1 = int(len(files)*valid_frac)
    split_index_2 = int(len(files)*(1-test_frac))

    # uncomment next line to get same split every time (necessary for time series)
    random.seed(1)
    random.shuffle(files)

    # validation split
    for file in files[:split_index_1]:
        if os.path.isfile(file):
            shutil.copy2(file, path_to_validation_tiles)

    # train split
    for file in files[split_index_1:split_index_2]:
        if os.path.isfile(file):
            shutil.copy2(file, path_to_train_tiles)

    # test split
    for file in files[split_index_2:]:
        if os.path.isfile(file):
            shutil.copy2(file, path_to_test_tiles)


if __name__ == '__main__':
    DATE = sys.argv[1]
    LEVEL = sys.argv[2]
    make_subset_split(DATE, LEVEL)
