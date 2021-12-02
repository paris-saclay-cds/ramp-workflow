.. _problem:

The problem.py file
###################

The ``problem.py`` file uses building blocks from the `RAMP workflow`_
library. These building blocks allow the ``problem.py`` file to be relatively
simple because the complexity is hidden by the implementation of the building
blocks in RAMP workflow. Titanic survival classification challenge
will be used as an example when discussing each aspect of the ``problem.py``
file. It is worth taking a look at the `whole file
<https://github.com/ramp-kits/titanic/blob/master/problem.py>`_ for reference.

1. The ``problem.py`` file begins by importing any libraries required. For
   example, the required libraries for the Titanic challenge were::

    import os
    import pandas as pd
    import rampwf as rw
    from sklearn.model_selection import StratifiedShuffleSplit

2. Provide a title::

    problem_title = 'Titanic survival classification'

3. Select a prediction type
   `Prediction types
   <https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/prediction_types>`_
   are classes used internally to store predictions and calculate scores. There
   are different prediction classes to be used with different types of
   challenges. This is implemented using functions, each of which returns a
   unique prediction class. The functions currently available are detailed
   below:

   * ``make_regression()`` - used for regression type challenges. Predictions
     can be 1-dimensional or 2-dimensional for multi-target regression. For
     multi-target regression a list of 'target names' needs to be provided
     to the parameter ``label_names``.
   * ``make_multiclass()`` - used for classification type challenges.
     Predictions are expected to be 2-dimensional (n samples * n classes) and
     may be labels or probabilities of each class. A list of label names needs
     to be provided to the parameter ``label_names``.
  * ``make_combined()`` - used for challenges where greater than one type of
    prediction is made. For example, the `drug spectra
    <https://github.com/ramp-kits/drug_spectra>`_
    challenge comprises of a classification task to predict the type of
    molecule and a regression task to predict the concentration of the molecule.
  * ``make_clustering()`` - used for clustering challenges. Prediction should
    be 2-dimensional (n samples * 2), where one column identifies the samples
    and the second column identifies the cluster it belongs to. This prediction
    class was used for the `High-energy physics tracking
    <https://github.com/ramp-kits/HEP_tracking>`_
    challenge.
  * ``make_detection()`` - a unique class specifically designed for the
    `Mars crater detection <https://github.com/ramp-kits/mars_craters>`_
    challenge. A unique algorithm is implemented for combining predictions
    from different models. See the `source code
    <https://github.com/paris-saclay-cds/ramp-workflow/blob/master/rampwf/prediction_types/detection.py>`_
    for more information.

  Select the appropriate prediction class for your challenge and state this
  in the ``problem.py`` file. If the appropriate prediction class does not
  exist in `RAMP workflow`_, you can define your own prediction class within
  the ``problem.py`` file. If it is not too specific, we would also encourage
  you to add your class to `RAMP workflow`_ so others can use it in future.
  See :ref:`contributing`.

  For example, the Titanic survival classification
  challenge aimed to predict whether or not each passenger survived. Survival
  is indicated by 0 (did not survive) or 1 (survived). Since this is a
  classification task the prediction function ``make_multiclass()`` is used.
  Note that by convention label names are stored as a variable
  ``_prediction_label_names``. The relevant parts of the ``problem.py`` file
  are shown below::

    _prediction_label_names = [0, 1]

    Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

4. Select a workflow
   `Workflows
   <https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/workflows>`_
   implement the steps of the machine learning problem. Each workflow
   is a class with a ``train_submission()`` and a ``test_submission()``
   function, which defines the workflow to implement during training and
   testing time. An attribute called ``workflow_element_names`` is also
   required. This attribute is a list of the file names that
   ``ramp-test`` expects for each submission. This class is
   implemented by RAMP workflow internals to train and test each submission of a
   challenge.

   * ``Estimator()`` - a versatile, commonly used workflow that will work for
     any `scikit-learn <https://scikit-learn.org/stable/>`_ estimator or
     pipeline (where the last step is an estimator). It imports a file
     named ``estimator.py`` from the submission directory (e.g.
     ``submissions/starting_kit/``). This ``estimator.py`` file should define a
     function named ``get_estimator()``, which returns a scikit-learn
     estimator or pipeline with a final estimator. The ``fit`` and ``predict``
     methods of the estimator/pipeline are then performed on the training
     and test data respectively.
  * ``EstimatorExternalData()`` - similar to the workflow above, except it
    allows an external data file, named``external_data.csv``, to be used in the
    workflow.

  There are also more specific workflows, for example for timeseries or image
  data. Some of specific workflows were designed for one specific challenge but
  can be re-used for similar challenges. The first four workflows below are used
  as building blocks for the more specific workflows that follow:

   * ``Regressor()`` - this workflow is for simple regression problems. It will
     import the file named ``regressor.py`` from the submission directory
     (e.g. ``submissions/starting_kit/``). This ``regressor.py`` file should
     define a class named ``Regressor`` that has a ``fit()`` and a ``predict()``
     method. This workflow will run ``fit()`` on the training data then at
     testing time, run ``predict()`` on the testing data, using the trained
     model. If you are using a scikit-learn function, you can simply call the
     ``fit()`` and ``predict()`` methods of the model you are using.
  * ``Classifier()`` - this workflow is for simple classification problems. It
    will import the file named ``classifier.py`` from the submission directory
    (e.g. ``submissions/starting_kit/``). This ``classifier.py`` file should
    define a class named ``Classifier`` that has a ``fit()`` and a
    ``predict_proba()`` method. This workflow will run ``fit()`` on the
    training data and at testing time, run ``predict_proba()`` on the test
    data, using the trained model. If you are using a scikit-learn function,
    you can simply call the ``fit()`` and ``predict_proba()`` methods of the
    model you are using.
  * ``FeatureExtractor()`` - this workflow is designed for preprocessing data,
    for example converting non-numeric features into numeric, normalising data
    and creating new features using existing features. It will import the file
    named ``feature_extractor.py`` from the submission directory
    (e.g. ``submissions/starting_kit/``). This ``feature_extractor.py`` file
    should define a class named ``FeatureExtractor`` with a ``fit()`` and a
    ``transform()`` method. This workflow will run ``fit()`` on the features
    and target of the data and run ``transform()`` on the features of training
    data. Note that ``fit()`` takes both the features and target of the data as
    input to enable feature engineering strategies such as target encoding
    during training time. The output of this workflow is the preprocessed
    features of the data.
  * ``feature_extractor_regressor()`` - this workflow combines the
    ``FeatureExtrator()`` and ``Regressor()`` workflows such that data is first
    preprocessed with ``FeatureExtractor()`` and then ``Regressor()``
    performs model training and prediction. Note that the ``fit()`` method of
    ``FeatureExtractor()`` is only performed on training data but not test
    data.
  * ``feature_extractor_classifier()`` - this workflow combines the
    ``FeatureExtractor()`` and ``Classifier()`` workflows such that data is
    first preprocessed with ``FeatureExtractor()`` and then ``Classifier()``
    performs model training and prediciton. As above the ``fit()`` method of
    ``FeatureExtractor()`` is only performed on training data but not test
    data.

  Workflows for specific data challenges:

  * ``ImageClassifier()`` - this workflow is for image classification
    tasks, particularly for cases when the dataset cannot be stored in memory.
    This workflow will import two files from the submissions folder;
    ``image_preprocessor.py`` and ``batch_classifier.py``.
    ``image_preprocessor.py`` should define a function called ``transform()``
    which preprocesses images. It should take an image as input and output an
    image. Optionally, this file can also define a function called
    ``transform_test()``, which is only used to preprocess images at test time.
    If this is not defined, ``transform()`` will be used at train and test time.
    ``batch_classifier.py`` should define a class called ``BatchClassifier``
    with the methods ``fit()`` and ``predict_prob()``. ``fit()`` should fit
    a model to batches of images (you can define batch size). For an example
    you can take a look at the `MNIST
    <https://github.com/ramp-kits/MNIST>`_
    or `Pollenating insects`_ challenges.
  * ``SimplifiedImageClassifier()`` - this is a simplified version of the
    above workflow where there is no image preprocessing step and instead of
    training and test batches of images, ``fit()`` and ``predict_proba()`` is
    performed on one image at a time. For an example, take a look at the
    `MNIST simplified <https://github.com/ramp-kits/MNIST_simplified>`_
    and `Pollenating insects`_ challenges.
  * ``ObjectDetector()`` - this workflow is used for image object detection
    tasks. It workflow imports one, ``object_detector.py``, from the
    submissions folder, which should define a class, ``ObjectDetector``, with
    ``fit()`` and ``predict()`` methods. It was used in the `Mars craters
    <https://github.com/ramp-kits/mars_craters>`_ challenge and the `Astronomy
    <https://github.com/ramp-kits/astrophd_tutorial>`_ tutorial.
  * ``Clusterer()`` - this workflow was used for the `High-energy physics
    tracking <https://github.com/ramp-kits/HEP_tracking>`_ challenge which
    aimed to cluster particle hits. This workflow
    imports the file named ``clusterer.py`` from the submissions directory.
    This file should define a class called  ``Clusterer`` with ``fit()``
    and ``predict_single_event()`` methods. ``fit()`` takes the
    features and the cluster ID of each sample as arguments to train the
    clustering model. At testing time, the each sample is sent to
    ``predict_single_event()`` separately and the predicted cluster assignments
    are joined with the sample ID (the first column of the features data) and
    returned.
  * ``ElNino()`` - this workflow was used for the `El Nino
    <https://github.com/ramp-kits/el_nino>`_ challenge which used temperature
    data over time to predict future temperatures. The workflow consists of
    the ``TimeSeriesFeatureExtractor()`` then ``Regressor()`` workflows.
  * ``GridFeatureExtractorClassifier()`` - this workflow was used in the
    `California rainfall <https://github.com/ramp-kits/california_rainfall>`_
    challenge. It consists of the ``GridFeatureExtractor()`` then
    ``Classifier()`` workflows. This workflow is similar to
    ``feature_extractor_classifier()`` except that ``GridFeatureExtractor()``
    takes as input 3 dimensional spatial grid data.
  * ``DrugSpectra()`` - this workflow was used for the `Drug spectra
    <https://github.com/ramp-kits/drug_spectra>`_ challenge. It implements
    both the ``feature_extractor_regressor()`` and
    ``feature_extractor_classifier()`` workflows to perform a classification
    task and a regression task on the same dataset. The submissions directory
    requires 4 files named; ``feature_extractor_clf.py``,
    ``classifier.py``, ``feature_extractor_reg.py`` and ``regressor.py``.

   If the appropriate workflow class does not exist in `RAMP workflow`_, you
   can define your own workflow class within the ``problem.py`` file. If it is
   not too specific,We would also encourage you to add your class to
   `RAMP workflow`_ so others can use it in future. See :ref:`contributing`.

   The Titanic challenge employed the ``feature_extractor_classifier()``
   workflow. This can be specified simply with::

    workflow = rw.workflows.FeatureExtractorClassifier()

.. _score-types:

5. Select score types
   Score types are metrics used to assess each submission. A large
   number of different `score metrics are available
   <https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/score_types>`_.
   To use one or more existing score metrics, simply provide a list of the
   class names of the score you wish to use and assign this to a variable
   called ``score_types``. For example, the Titanic challenge used 3
   different score metrics::

    score_types = [
        rw.score_types.ROCAUC(name='auc'),
        rw.score_types.Accuracy(name='acc'),
        rw.score_types.NegativeLogLikelihood(name='nll'),
    ]

   If you select more than one score, all the score metrics will be calculated
   when you enter a submission to RAMP. You can select one score metric to be
   used as the official score, used to rank participants, or calculate a
   weighted combined score from the various score metrics. For example, the
   `Drug spectra <https://github.com/ramp-kits/drug_spectra>`_
   challenge used a weighted combination of ``ClassificationError`` and
   ``MARE`` (Mean Absolute Relative Error)::

    score_type_1 = rw.score_types.ClassificationError(name='err', precision=3)
    score_type_2 = rw.score_types.MARE(name='mare', precision=3)
    score_types = [
        # The official score combines the two scores with weights 2/3 and 1/3.
        rw.score_types.Combined(
            name='combined', score_types=[score_type_1, score_type_2],
            weights=[2. / 3, 1. / 3], precision=3),
    ]

  Note that the actual implementation was more complex as this challenge
  consisted of both a classification and regression task. For the purposes of
  this example, the extra complexity was ignored.

  Again if the appropriate score metric class does not exist in
  `RAMP workflow`_, you can define your own score metric class within the
  ``problem.py`` file. If it is not too specific, we would also encourage you
  to add your class to `RAMP workflow`_ so others can use it in future. See
  :ref:`contributing`.

.. _cross-validation:

6. Specify a cross-validation scheme
   Specify a way to split the 'train' data into training and validation sets.
   This should be done by defining a ``get_cv()`` function that takes
   the feature and target data as parameters and returns indicies that can
   be used to split the data. If you are using a function with a random
   element, e.g., ``StratifiedShuffleSplit()`` `from scikit-learn
   <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit.split>`_,
   it is important to set the random seed. This ensures that the train and
   valuidation data will be the same for all participants.

   For example, the Titanic challenge used ``StratifiedShuffleSplit()``::

    def get_cv(X, y):
        cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
        return cv.split(X, y)

.. _in-out:

7. Provide the I/O methods
   The ``problem.py`` file needs to define a ``get_train_data()`` and a
   ``get_test_data()`` function that reads in the training and test data. These
   functions will be used to 'get data' both locally and on the RAMP sever. For
   example, this was implemented in the Titanic challenge using::

        _target_column_name = 'Survived'
        _ignore_column_names = ['PassengerId']

        def _read_data(path, f_name):
            data = pd.read_csv(os.path.join(path, 'data', f_name))
            y_array = data[_target_column_name].values
            X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
            return X_df, y_array


        def get_train_data(path='.'):
            f_name = 'train.csv'
            return _read_data(path, f_name)


        def get_test_data(path='.'):
            f_name = 'test.csv'
            return _read_data(path, f_name)

   The ``_read_data()`` is not strictly required and is acting as a helper
   function in the code above.

8. If the ``problem.py`` file becomes too long and you would like to refactor
   it, you can add an ``external_imports`` folder in the ``ramp_kit_dir`` and
   have modules there that you can import from. The way this works is that
   running ``ramp-test`` adds the ``external_imports`` folder to ``sys.path``
   if such a folder exists. For instance if you have a ``utils.py`` module
   in the ``external_imports`` folder::

        external_imports/
            utils.py

   Then you can do ``import utils`` in ``problem.py``.

.. _RAMP workflow: https://github.com/paris-saclay-cds/ramp-workflow
.. _Pollenating insects: <https://github.com/ramp-kits/pollenating_insects>`_
