"""
This file contains several dictionaries that are used throughout various scripts.
"""

import numpy as np

combinations_dict = {
    0: ['20180330', '20180419', '20180703', '20180807', '20180926', '20181205'],
    1: ['20180330', '20180529', '20180703', '20180807', '20180926', '20181205'],
    2: ['20180419', '20180529', '20180703', '20180807', '20180926', '20181205'],
    3: ['20180330', '20180419', '20180703', '20180807', '20181031', '20181205'],
    4: ['20180330', '20180529', '20180703', '20180807', '20181031', '20181205'],
    5: ['20180419', '20180529', '20180703', '20180807', '20181031', '20181205']
}

inv_category_dict = {
        1: 'Buildings',
        2: 'Grassland',
        3: 'Forest / Wood',
        4: 'Settlement Area without Buildings',
        5: 'Farmland',
        6: 'Industry and Commerce',
        7: 'Water Body',
        8: 'Unclassifiable',
        9: 'Change to Ground Truth',
        0: None
}

color_dict = {
    'Outside Area of Interest': np.array([0., 0., 0., 1.0]),
    'Buildings': np.array([1.0, 0., 1.0, 1.0]),
    'Grassland': np.array([0.48, 0.98, 0.0, 1.0]),
    'Forest / Wood': np.array([0.13, 0.54, 0.13, 1.0]),
    'Settlement Area without Buildings': np.array([1.0, 1.0, 0.31, 1.0]),
    'Farmland': np.array([0.803921568627451, 0.52, 0.24, 1.0]),
    'Industry and Commerce': np.array([0.50, 0.50, 0.50, 1.0]),
    'Water Body': np.array([0.0, 0.74, 1.0, 1.0]),
    'Unclassifiable': np.array([1.0, 0.0, 0.0, 1.0]),
    'Change to Ground Truth': np.array([1.0, 1.0, 1.0, 1.0]),
}
