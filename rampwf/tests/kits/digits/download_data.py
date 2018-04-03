from __future__ import division

import os

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from skimage.io import imsave

KIT_DIR = os.path.dirname(__file__)


def fetch_data():
    # create locally the images to load them afterwards
    digits = load_digits()

    data_dir = os.path.join(KIT_DIR, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    n_images = digits.data.shape[0]
    n_train_images = int(n_images * 0.8)

    img_dir = os.path.join(data_dir, 'imgs')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    filenames_image = []
    for img_idx, img in zip(range(n_images), digits.data):
        filename = os.path.join(img_dir, str(img_idx) + '.png')
        imsave(filename, img.reshape((8, 8)).astype(np.int8))
        filenames_image.append(filename)

    train_csv = pd.DataFrame({'id': np.array(filenames_image[:n_train_images]),
                              'class': digits.target[:n_train_images]})
    train_csv = train_csv.set_index('id')
    train_csv.to_csv(os.path.join(data_dir, 'train.csv'))
    test_csv = pd.DataFrame({'id': np.array(filenames_image[n_train_images:]),
                             'class': digits.target[n_train_images:]})
    test_csv = test_csv.set_index('id')
    test_csv.to_csv(os.path.join(data_dir, 'test.csv'))
