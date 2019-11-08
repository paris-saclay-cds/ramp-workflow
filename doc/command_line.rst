.. _command-line:

RAMP workflow commands
######################

``ramp_test_submission``
^^^^^^^^^^^^^^^^^^^^^^^^

Tests your RAMP submission. Has no required arguments and will by default test
the submission: ``./submissions/starting_kit/``.

Options:

* ``[--ramp_kit_dir]`` Root directory of the 'ramp-kit' to test. Default:
  ``'.'``.
* ``[--ramp_data_dir]`` Directory containing the data. This directory should
  contain a 'data' folder. Default: ``'.'``.
* ``[--ramp_submission_dir]`` Directory where the submissions are stored.
  Default: ``'.'``.                            
* ``[--submission]`` The folder name of the submission to test. It should be
  located in the ``ramp_submission_dir``. If ``'ALL'``, all submissions in the
  directory will be tested. Default: ``starting_kit``.
* ``[--quick-test]`` Specify this flag to test the submission on a small subset
  of the data.
* ``[--pickle]`` Specify this flag to pickle the submission after training.
* ``[--save-output]`` Specify this flag to save predictions, scores, eventual
  error trace, and state after training.
* ``[--retrain]`` Specify this flag to retrain the submission on the full
  training set after the CV loops.

``ramp_blend_submissions``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Takes a list of submissions and finds the best 'blend' of submissions that
result in the best combined score. See :ref:`scoring` for more details on how
predictions are combined to calculate the combined score. The relative
weighting of each submission, the overall combined score and best combined
score for each fold is output.

Options:

* ``[--ramp_kit_dir]`` Root directory of the ramp-kit to test. Default: ``'.'``.
* ``[--ramp_data_dir]`` Directory containing the data. This directory should
  contain a 'data' folder. Default: ``'.'``.
* ``[--ramp_submission_dir]`` Directory where the submissions are stored.
  Default: ``.``.
* ``[--submission]`` The folder name of the submission to blend. They should be
  located in the ``ramp_submission_dir``. Separate submissions by a comma
  without any spaces. If ``'ALL'``, all submissions in the 
  ``ramp_submission_dir`` directory will be blended. Default: ``'ALL'``.
* ``[--save_output]`` Specify this flag to save predictions after blending.
* ``[--min-improvement]`` The minimum score improvement when adding submissions
  to the ensemble. Default: ``0.0``.

``ramp_leaderboard``
^^^^^^^^^^^^^^^^^^^^

Pretty prints scores from previous locally tested submissions (using
``ramp_test_submission``). Previous submissions need to be saved using
``ramp_test_submission --submission <name> --save-output``.

Options:

* ``[--ramp_kit_dir]`` Root directory of the RAMP-kit. Default ``'.'``.
* ``[--cols]`` list of columns (separated by ',') to display. By default it is
  'train\_<metric>,valid\_<metric>,test\_<metric>' where <metric> is the first
  score metric from ``score_types`` (see :ref:`score types <score-types>`)
  according to
  alphabetical order. Use ``--help-cols`` to show the column names. Column
  names are of the form '<step>_<metric>' where <step> could be one of 'train',
  'valid' or 'test' and <metric> is any ``score_type`` defined in
  ``problem.py``. Default: ``None``.
* ``[--asc]`` sort scores in ascending order if ``True``, otherwise descending
  order. Default: ``False``.
* ``[--metric]`` metric to display. This can be used instead of specifying
  ``--cols``. For example, ``--metric=acc`` is equivalent to
  ``--cols=train_acc,valid_acc,test_acc``. Default: ``None``.
* ``[--precision``] precision for rounding. Default: ``2``.

Help:

* ``[--help-cols]`` prints the list of column names.
* ``[--help-metrics]`` prints the list of score metrics.

Examples:

* ``ramp_leaderboard --metric=acc`` 
* ``ramp_leaderboard --cols=train_acc,valid_acc,test_acc``
* ``ramp_leaderboard --cols=train_nll --sort-by=train_nll,train_acc --asc``

**Commands for use on server**

``ramp_test_notebook``
^^^^^^^^^^^^^^^^^^^^^^

First converts the starting kit notebook into HTML using ``nbconvert`` then
tests if the notebook can be executed.

Options:

* ``[--ramp_kit_dir]`` Directory containing the notebook. Default: ``'.'``.

``ramp_convert_notebook``
^^^^^^^^^^^^^^^^^^^^^^^^^

Converts the starting kit notebook into HTML using ``nbconvert``. This is would
be used on the welcome page of the challenge on RAMP studio.

Options:

* ``[--ramp_kit_dir]`` Directory containing the notebook. Default: ``'.'``.
 