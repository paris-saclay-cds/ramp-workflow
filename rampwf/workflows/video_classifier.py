from __future__ import division

import os

from glob import glob

from ..utils.importing import import_file


class VideoClassifier(object):
    """
    VideoClassifier workflow.

    This workflow is used to train video classification tasks.
    It is the video version of SimplifiedImageClassifier workflow.
    Submissions need to contain one file, which by default by is named
    video_classifier.py (it can be modified by changing
    `workflow_element_names`).
    video_classifier.py needs a `VideoClassifier` class, which implements
    `fit` and `predict_proba`, where both `fit` and `predict_proba` take
    as input an instance of `VideoLoader`.

    Parameters
    ==========

    label_to_id : dict
        Convert a label name (str) to an integer
    """

    def __init__(self, label_to_id,
                 workflow_element_names=['video_classifier']):
        self.label_to_id = label_to_id
        self.element_names = workflow_element_names

    def train_submission(self, module_path, folder_X_array, y_array,
                         train_is=None):
        """Train a video classifier.

        module_path : str
            module where the submission is. the folder of the module
            have to contain video_classifier.py.
        X_array : vector of int or str
            vector of videos IDs to train on
        y_array : vector of int or str
            vector of video labels corresponding to X_train
        train_is : vector of int
           indices from X_array to train on
        """
        folder, X_array = folder_X_array
        if train_is is None:
            train_is = slice(None, None, None)
        video_classifier = import_file(module_path, self.element_names[0])
        clf = video_classifier.VideoClassifier()
        video_loader = VideoLoader(
            X_array[train_is], y_array[train_is],
            folder=folder,
            label_to_id=self.label_to_id)

        clf.fit(video_loader)
        return clf

    def test_submission(self, trained_model, folder_X_array):
        """Test an video classifier.

        trained_model : tuple (function, Classifier)
            tuple of a trained model returned by `train_submission`.
        X_array : vector of int or str
            vector of video IDs to test on.
        """
        folder, X_array = folder_X_array
        clf = trained_model
        test_video_loader = VideoLoader(
            X_array, None,
            folder=folder,
            label_to_id=self.label_to_id,
        )
        y_proba = clf.predict_proba(test_video_loader)
        return y_proba


class VideoLoader(object):
    """
    Load video frames and their associated label

    In video_classifier.py, both `fit` and `predict_proba` take as input
    an instance of `VideoLoader`.
    Video frames are loaded by using the method `load`.

    Parameters
    ==========

    X_array : vector of int or str
         vector of video IDs to train on

    y_array : vector of int or vector of str or None
        vector of video labels corresponding to `X_array`.
        At test time, it is `None`.

    folder : str
        folder where the video frames are

    n_classes : int
        Total number of classes.
    """

    def __init__(self, X_array, y_array, folder, label_to_id):
        self.X_array = X_array
        self.y_array = y_array
        self.folder = folder
        self.label_to_id = label_to_id
        self.nb_examples = len(X_array)

    def load(self, index, frame_id):
        """
        Load either one frame from a video and the label corresponding
        to the video or one frame from a video (at test time)
        or one image (at test time).

        Parameters
        ==========

        index : int
            Index of the video to load.
            It should in between 0 and self.nb_examples - 1

        Returns
        =======

        either a tuple `(x, y)` or `x`, where:
            - x is a numpy array of shape (height, width, nb_color_channels),
              and corresponds to the image of the frame `frame_id` of the
              video `index`.
            - y is an integer, corresponding to the class of the video `index`.
        At training time, `y_array` is given, and `load` returns
        a tuple (x, y).
        At test time, `y_array` is `None`, and `load` returns `x`.
        """
        from skimage.io import imread

        if index < 0 or index >= self.nb_examples:
            raise IndexError("list index out of range")
        if frame_id < 0 or frame_id >= self.nb_frames(index):
            raise IndexError("frame index out of range")
        frame_id += 1  # frames start by 1
        x = self.X_array[index]
        filename = os.path.join(
            self.folder,
            '{}'.format(x),
            'image_{:010d}.jpg'.format(frame_id)
        )
        x = imread(filename)
        if self.y_array is not None:
            y = self.y_array[index]
            return x, self.label_to_id[y]
        else:
            return x

    def nb_frames(self, index):
        """
        Returns the total number of frames of a video
        """
        if index < 0 or index >= self.nb_examples:
            raise IndexError("list index out of range")
        # when nb of frames is 0 it means that the video
        # could not be downloaded correctly. Some videos
        # in the kinetics dataset have been removed by the "user"
        # in youtube.
        x = self.X_array[index]
        folder = os.path.join(
            self.folder,
            '{}'.format(x),
            '*.jpg'
        )
        return len(glob(folder))

    def parallel_load(self, indexes, transforms=None):
        """
        Parallel version of load.

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
        from skimage.io import imread
        from joblib import delayed, Parallel, cpu_count

        for index in indexes:
            assert 0 <= index < self.nb_examples

        n_jobs = cpu_count()
        filenames = [
            os.path.join(self.folder, '{}'.format(self.X_array[index]))
            for index in indexes]
        xs = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(imread)(filename) for filename in filenames)

        if self.y_array is not None:
            ys = [self.y_array[index] for index in indexes]
            return xs, self.label_to_id[ys]
        else:
            return xs

    def __iter__(self):
        for i in range(self.nb_examples):
            yield self.load(i)

    def __len__(self):
        return self.nb_examples
