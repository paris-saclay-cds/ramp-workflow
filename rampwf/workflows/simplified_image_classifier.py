from __future__ import division
import os
import imp
import numpy as np
from skimage.io import imread


class SimplifiedImageClassifier(object):
    """
    SimplifiedImageClassifier workflow.
    This workflow is used to train image classification tasks, typically when
    the dataset cannot be stored in memory. It is a simplified version
    of the `ImageClassifier` workflow where there is no batch generator
    and no image preprocessor. 
    Submissions need to contain one file, which by default by is named
    image_classifier.py (it can be modified by changing 
    `workflow_element_names`).
    image_classifier.py needs an `ImageClassifier` class, which implements
    `fit` and `predict_proba`, where both `fit` and `predict_proba` take 
    as input an instance of `ImageLoader`.

    Parameters
    ==========

    n_classes : int
        Total number of classes.

    """
    def __init__(self, n_classes, workflow_element_names=['image_classifier']):
        self.n_classes = n_classes
        self.element_names = workflow_element_names

    def train_submission(self, module_path, folder_X_array, y_array,
                         train_is=None):
        """Train an image classifier.

        module_path : str
            module where the submission is. the folder of the module
            have to contain image_classifier.py.
        X_array : ArrayContainer vector of int
            vector of image IDs to train on
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        y_array : vector of int
            vector of image labels corresponding to X_train
        train_is : vector of int
           indices from X_array to train on
        """
        folder, X_array = folder_X_array
        if train_is is None:
            train_is = slice(None, None, None)
        submitted_image_classifier_file = '{}/{}.py'.format(
            module_path, self.element_names[0])
        image_classifier = imp.load_source('', submitted_image_classifier_file)
        clf = image_classifier.ImageClassifier()
        img_loader = ImageLoader(
            X_array[train_is], y_array[train_is],
            folder=folder,
            n_classes=self.n_classes)
        clf.fit(img_loader)
        return clf

    def test_submission(self, trained_model, folder_X_array):
        """Test an image classifier.

        trained_model : tuple (function, Classifier)
            tuple of a trained model returned by `train_submission`.
        X_array : ArrayContainer of int
            vector of image IDs to test on.
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        """
        folder, X_array = folder_X_array
        clf = trained_model
        test_img_loader = ImageLoader(
            X_array, None,
            folder=folder,
            n_classes=self.n_classes
        )
        y_proba = clf.predict_proba(test_img_loader)
        return y_proba


class ImageLoader(object):
    """   
    In image_classifier.py, both `fit` and `predict_proba` take as input
    an instance of `ImageLoader`.
    ImageLoader is used in `fit` and `predict_proba` to either load one image 
    and its corresponding label  (at training time), or one image (at test time).
    Images are loaded by using the method `load`.

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image IDs to train on
         (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).

    y_array : vector of int or None
        vector of image labels corresponding to `X_array`.
        At test time, it is `None`.

    folder : str
        folder where the images are

    n_classes : int
        Total number of classes. 
    """

    def __init__(self, X_array, y_array, folder, n_classes):
        self.X_array = X_array
        self.y_array = y_array
        self.folder = folder
        self.n_classes = n_classes
        self.nb_examples = len(X_array)
    
    def load(self, index):
        """
        
        load one image and its corresponding label (at training time),
        or one image (at test time).

        Parameters 
        ==========

        index : int
            Index of the image to load.
            It should in between 0 and self.nb_examples - 1
    
        Returns
        =======

        either a tuple `(x, y)` or `x`, where:
            - x is a numpy array of shape (height, width, nb_color_channels),
              and corresponds to the image of the requested `index`.
            - y is an integer, corresponding to the class of `x`.
        At training time, `y_array` is given, and `load` returns
        a tuple (x, y).
        At test time, `y_array` is `None`, and `load` returns `x`.
        """
        assert 0 <= index < self.nb_examples

        x = self.X_array[index]
        filename = os.path.join(self.folder, '{}'.format(x))
        x = imread(filename)
        if self.y_array is not None:
            y = self.y_array[index]
            return x, y
        else:
            return x

    def __iter__(self):
        for i in range(self.nb_examples):
            yield self.load(i)

    def __len__(self):
        return self.nb_examples
