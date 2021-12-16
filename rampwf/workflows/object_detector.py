import os

import numpy as np

from ..utils.importing import import_module_from_source


class ObjectDetector(object):
    """
    Object detection workflow.

    This workflow is used to train image object detection tasks, typically
    when the dataset cannot be stored in memory.
    Submissions need to contain two files, which by default are named:
    image_preprocessor.py and object_detector_model.py (they can be
    modified by changing `workflow_element_names`).

    image_preprocessor.py needs a `transform` function, which
    is used for preprocessing the images. It takes an image as input
    and it returns an image as an output. Optionally, image_preprocessor.py
    can also have a function `transform_test`, which is used only to preprocess
    images at test time. Otherwise, if `transform_test` does not exist,
    `transform` is used at train and test time.

    object_detector_model.py needs a `ObjectDetector` class, which
    implements `fit` and `predict`.

    Parameters
    ==========

    test_batch_size : int
        batch size used for testing.

    chunk_size : int
        size of the chunk used to load data from disk into memory.
        (see at the top of the file what a chunk is and its difference
         with the mini-batch size of neural nets).

    n_jobs : int
        the number of jobs used to load images from disk to memory as `chunks`.

    """

    def __init__(self, workflow_element_names=['object_detector']):
        self.element_names = workflow_element_names

    def train_submission(self, module_path, X, y, train_is=None):
        """Train a ObjectDetector.

        module_path : str
            module where the submission is. the folder of the module
            have to contain object_detector.py.
        X : ArrayContainer vector of int
            vector of image data to train on
        y : vector of lists
            vector of object labels corresponding to X
        train_is : vector of int
           indices from X_array to train on
        """
        if train_is is None:
            train_is = slice(None, None, None)

        # object detector model
        detector = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        clf = detector.ObjectDetector()

        # train and return fitted model
        clf.fit(X[train_is], y[train_is])
        return clf

    def test_submission(self, trained_model, X):
        """Test an ObjectDetector.

        trained_model
            Trained model returned by `train_submission`.
        X : ArrayContainer of int
            Vector of image data to test on.
        """
        clf = trained_model
        y_pred = clf.predict(X)
        return y_pred


class BatchGeneratorBuilder(object):
    """A batch generator builder for generating batches of images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).

    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image data to train on
    y_array : vector of int
        vector of object labels corresponding to `X_array`

    """

    def __init__(self, X_array, y_array):
        self.X_array = X_array
        self.y_array = y_array
        self.nb_examples = len(X_array)

    def get_train_valid_generators(self, batch_size=256, valid_ratio=0.1):
        """Build train and valid generators for keras.

        This method is used by the user defined `Classifier` to o build train
        and valid generators that will be used in keras `fit_generator`.

        Parameters
        ==========

        batch_size : int
            size of mini-batches
        valid_ratio : float between 0 and 1
            ratio of validation data

        Returns
        =======

        a 4-tuple (gen_train, gen_valid, nb_train, nb_valid) where:
            - gen_train is a generator function for training data
            - gen_valid is a generator function for valid data
            - nb_train is the number of training examples
            - nb_valid is the number of validation examples
        The number of training and validation data are necessary
        so that we can use the keras method `fit_generator`.
        """
        nb_valid = int(valid_ratio * self.nb_examples)
        nb_train = self.nb_examples - nb_valid
        indices = np.arange(self.nb_examples)
        train_indices = indices[0:nb_train]
        valid_indices = indices[nb_train:]
        gen_train = self._get_generator(
            indices=train_indices, batch_size=batch_size)
        gen_valid = self._get_generator(
            indices=valid_indices, batch_size=batch_size)
        return gen_train, gen_valid, nb_train, nb_valid

    def _get_generator(self, indices=None, batch_size=256):
        if indices is None:
            indices = np.arange(self.nb_examples)
        # Infinite loop, as required by keras `fit_generator`.
        # However, as we provide the number of examples per epoch
        # and the user specifies the total number of epochs, it will
        # be able to end.
        while True:
            X = self.X_array[indices]
            y = self.y_array[indices]

            # converting to float needed?
            # X = np.array(X, dtype='float32')

            # Yielding mini-batches
            for i in range(0, len(X), batch_size):
                yield X[i:i + batch_size], y[i:i + batch_size]
