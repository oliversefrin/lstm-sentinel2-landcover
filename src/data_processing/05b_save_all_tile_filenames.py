"""
Save filenames of all tiles in a .txt file.

This lookup .txt is needed when building sequences of multiple images.
"""

from glob import glob

# select a date of a complete satellite image, here 2016-12-06
date = '20161206'
# get all file endings
filenames = [x[-12:] for x in
             glob(f'../../data/processed/training_data/{date}_data/{date}_L1C_all_tiles_(32x32)/*.tif')]
# save them to file
with open('../../data/processed/all_tile_endings.txt', 'w') as file:
    for filename in filenames:
        file.write(filename + '\n')
