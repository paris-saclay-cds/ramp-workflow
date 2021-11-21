import os

import numpy as np

from ..utils.importing import import_module_from_source


class ImageClassifier(object):
    """
    ImageClassifier workflow.

    This workflow is used to train image classification tasks, typically when
    the dataset cannot be stored in memory.
    Submissions need to contain two files, which by default are named:
    image_preprocessor.py and batch_classifier.py (they can be modified
    by changing `workflow_element_names`).

    image_preprocessor.py needs a `tranform` function, which
    is used for preprocessing the images. It takes an image as input
    and it returns an image as an output. Optionally, image_preprocessor.py
    can also have a function `transform_test`, which is used only to preprocess
    images at test time. Otherwise, if `transform_test` does not exist,
    `transform` is used at train and test time.

    batch_classifier.py needs a `BatchClassifier` class, which implements
    `fit` and `predict_proba`, where `fit` takes as input an instance
    of `BatchGeneratorBuilder`.

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

    n_classes : int
        Total number of classes.

    """

    def __init__(self, test_batch_size, chunk_size, n_jobs, n_classes,
                 workflow_element_names=[
                     'image_preprocessor', 'batch_classifier']):
        self.element_names = workflow_element_names
        self.test_batch_size = test_batch_size
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.n_classes = n_classes

    def train_submission(self, module_path, folder_X_array, y_array,
                         train_is=None):
        """Train a batch image classifier.

        module_path : str
            module where the submission is. the folder of the module
            have to contain batch_classifier.py and image_preprocessor.py.
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
        image_preprocessor = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        transform_img = image_preprocessor.transform
        transform_test_img = getattr(image_preprocessor,
                                     'transform_test',
                                     transform_img)
        batch_classifier = import_module_from_source(
            os.path.join(module_path, self.element_names[1] + '.py'),
            self.element_names[1],
            sanitize=True
        )
        clf = batch_classifier.BatchClassifier()

        gen_builder = BatchGeneratorBuilder(
            X_array[train_is], y_array[train_is],
            transform_img, transform_test_img,
            folder=folder,
            chunk_size=self.chunk_size, n_classes=self.n_classes,
            n_jobs=self.n_jobs)
        clf.fit(gen_builder)
        return transform_img, transform_test_img, clf

    def test_submission(self, trained_model, folder_X_array):
        """Train a batch image classifier.

        trained_model : tuple (function, Classifier)
            tuple of a trained model returned by `train_submission`.
        X_array : ArrayContainer of int
            vector of image IDs to test on.
            (it is named X_array to be coherent with the current API,
             but as said here, it does not represent the data itself,
             only image IDs).
        """
        folder, X_array = folder_X_array
        transform_img, transform_test_img, clf = trained_model
        it = _chunk_iterator(
            X_array, folder=folder, chunk_size=self.chunk_size)
        y_proba = []
        for X in it:
            for i in range(0, len(X), self.test_batch_size):
                # 1) Preprocessing
                X_batch = X[i: i + self.test_batch_size]
                # X_batch = Parallel(n_jobs=self.n_jobs, backend='threading')(
                #     delayed(transform_img)(x) for x in X_batch)
                X_batch = [transform_test_img(x) for x in X_batch]
                # X is a list of numpy arrays at this point, convert it to a
                # single numpy array.
                try:
                    X_batch = [x[np.newaxis, :, :, :] for x in X_batch]
                except IndexError:
                    # single channel
                    X_batch = [
                        x[np.newaxis, np.newaxis, :, :] for x in X_batch]
                X_batch = np.concatenate(X_batch, axis=0)

                # 2) Prediction
                y_proba_batch = clf.predict_proba(X_batch)
                y_proba.append(y_proba_batch)
        y_proba = np.concatenate(y_proba, axis=0)
        return y_proba


class BatchGeneratorBuilder(object):
    """A batch generator builder for generating images on the fly.

    This class is a way to build training and
    validation generators that yield each time a tuple (X, y) of mini-batches.
    The generators are built in a way to fit into keras API of `fit_generator`
    (see https://keras.io/models/model/).
    An instance of this class is exposed to users `Classifier` through
    the `fit` function : model fitting is called by using
    "clf.fit(gen_builder)" where `gen_builder` is an instance
    of this class : `BatchGeneratorBuilder`.
    The fit function from `Classifier` should then use the instance
    to build train and validation generators, using the method
    `get_train_valid_generators`

    Parameters
    ==========

    X_array : ArrayContainer of int
        vector of image IDs to train on
         (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).

    y_array : vector of int
        vector of image labels corresponding to `X_array`

    folder : str
        folder where the images are

    chunk_size : int
        size of the chunk used to load data from disk into memory.
        (see at the top of the file what a chunk is and its difference
         with the mini-batch size of neural nets).

    n_classes : int
        Total number of classes. This is needed because the array
        of labels, which is a vector of ints, is transformed into
        a onehot representation.

    n_jobs : int
        the number of jobs used to load images from disk to memory as `chunks`.
    """

    def __init__(self, X_array, y_array,
                 transform_img, transform_test_img,
                 folder, chunk_size, n_classes, n_jobs):
        self.X_array = X_array
        self.y_array = y_array
        self.transform_img = transform_img
        self.transform_test_img = transform_test_img
        self.folder = folder
        self.chunk_size = chunk_size
        self.n_classes = n_classes
        self.n_jobs = n_jobs
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
            it = _chunk_iterator(
                X_array=self.X_array[indices], folder=self.folder,
                y_array=self.y_array[indices], chunk_size=self.chunk_size,
                n_jobs=self.n_jobs)
            for X, y in it:
                # 1) Preprocessing of X and y
                # X = Parallel(
                # n_jobs=self.n_jobs, backend='threading')(delayed(
                #     self.transform_img)(x) for x in X)
                X = np.array([self.transform_img(x) for x in X])
                # # X is a list of numpy arrays at this point, convert it to a
                # single numpy array.
                try:
                    X = [x[np.newaxis, :, :, :] for x in X]
                except IndexError:
                    # single channel
                    X = [x[np.newaxis, np.newaxis, :, :] for x in X]
                X = np.concatenate(X, axis=0)
                X = np.array(X, dtype='float32')
                # Convert y to onehot representation
                y = _to_categorical(y, num_classes=self.n_classes)

                # 2) Yielding mini-batches
                for i in range(0, len(X), batch_size):
                    yield X[i:i + batch_size], y[i:i + batch_size]


def _chunk_iterator(X_array, folder, y_array=None, chunk_size=1024, n_jobs=8):
    """Generate chunks of images, optionally with their labels.

    Parameters
    ==========

    X_array : ArrayContainer of int
        image ids to load
        (it is named X_array to be coherent with the current API,
         but as said here, it does not represent the data itself,
         only image IDs).

    y_array : vector of int
        labels corresponding to each image from X_array

    chunk_size : int
        chunk size

    folder : str
        folder where the images are

    n_jobs : int
        number of jobs used to load images in parallel

    Yields
    ======

    if y_array is provided (not None):
        it yields each time a tuple (X, y) where X is a list
        of numpy arrays of images and y is a list of ints (labels).
        The length of X and y is `chunk_size` at most (it can be smaller).

    if y_array is not provided (it is None)
        it yields each time X where X is a list of numpy arrays
        of images. The length of X is `chunk_size` at most (it can be
        smaller).
        This is used for testing, where we don't have/need the labels.

    The shape of each element of X in both cases
    is (height, width, color), where color is 1 or 3 or 4 and height/width
    vary according to examples (hence the fact that X is a list instead of
    numpy array).
    """
    from skimage.io import imread
    from joblib import delayed
    from joblib import Parallel
    for i in range(0, len(X_array), chunk_size):
        X_chunk = X_array[i:i + chunk_size]
        filenames = [
            os.path.join(folder, '{}'.format(x))
            for x in X_chunk]
        X = Parallel(n_jobs=n_jobs, backend='threading')(delayed(imread)(
            filename) for filename in filenames)
        if y_array is not None:
            y = y_array[i:i + chunk_size]
            yield X, y
        else:
            yield X


def _to_categorical(y, num_classes=None):
    """Convert a class vector (integers) to binary class matrix.

    Taken from keras:
    https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py
    The reason it was taken from keras is to avoid importing theano which
    clashes with pytorch.

    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def get_nb_minibatches(nb_samples, batch_size):
    """Compute the number of minibatches for keras.

    See [https://keras.io/models/sequential]
    """
    return (nb_samples // batch_size) +\
        (1 if (nb_samples % batch_size) > 0 else 0)
