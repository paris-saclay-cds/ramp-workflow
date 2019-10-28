.. _scoring:

RAMP scoring
############

Local submissions
=================

When testing a submission locally (i.e. with ``ramp_test_submission``) a number
of scores will be calculated and printed to standard output. The scores will
look like this::

    Testing Titanic survival classification
    Reading train and test files from ./data ...
    Reading cv ...
    Training ./submissions/random_forest_20_5 ...
    CV fold 0
        train auc = 0.84
        valid auc = 0.89
        test auc = 0.83
    CV fold 1
        train auc = 0.85
        valid auc = 0.86
        test auc = 0.83
    CV fold 2
        train auc = 0.85
        valid auc = 0.83
        test auc = 0.82
    CV fold 3
        train auc = 0.84
        valid auc = 0.91
        test auc = 0.83
    CV fold 4
        train auc = 0.85
        valid auc = 0.87
        test auc = 0.83
    CV fold 5
        train auc = 0.84
        valid auc = 0.89
        test auc = 0.84
    CV fold 6
        train auc = 0.84
        valid auc = 0.88
        test auc = 0.84
    CV fold 7
        train auc = 0.85
        valid auc = 0.86
        test auc = 0.84
    ----------------------------
    Mean CV scores
    ----------------------------
        train auc = 0.85 ± 0.005
        valid auc = 0.87 ± 0.023
        test auc = 0.83 ± 0.006
    ----------------------------
    Bagged scores
    ----------------------------
        score   auc
        valid  0.875
        test   0.834

Locally, there should be a training dataset and a testing dataset, usually
within a folder named ``data/``. We will call these datasets the 'public'
training data and the 'public' test data. This is because, for a RAMP event,
there will also be private training and test data (see :ref:`data` for more).

Eight-fold cross-validation is performed, whereby the public training data is
split into 'training' and 'validation' subsets 8 times. The subsets are
different each time. For each cross-validation fold, the model is trained with
the training data and used to predict targets for the validation subset and the
public testing data. The scores are computed for the training, validation and
testing datasets, for each fold. The mean of these 8 scores are calculated and
printed under ``Mean CV scores``. In the example above, there is only one
score metric 'auc'. If more than one score metric was defined in ``problem.py``
(see :ref:`score-types`), scores for all the score metrics will be printed.

``Bagged scores`` are calculated by combining the predictions of the 8 folds
and using the combined prediction to calculate the score. For regression
problems the combined prediction is just the mean of the predictions and
for classification problems, it is the mean probability of each class. For
detection problems the combined prediction calculation is more complex. See
the `source code 
<https://github.com/paris-saclay-cds/ramp-workflow/blob/12512a3192bcc515c2da956a6a6704849cdadeee/rampwf/prediction_types/detection.py#L37>`_
for more details.

For example, the Titanic challenge aims to predict whether or no each
passenger survived. For each fold, different survival predictions are made for
the test data. This is because each model is different as it was trained using
different data). The probality of each classification (survived or did not
survive), from the 8 predictions, is averaged and the classification computed
with the new average probabilities. This is done for each sample in the test
data and the new 'combined prediction' is used to calculate the 'bagged' score.
This differs slightly for the validation score because the validation datasets
will be different between each fold, and samples may or may not overlap between
folds. In cases where there is only one prediction for a validation sample, the
combined prediction will simply be the single prediction.

Note that technically this is not what bagging means, but the name is used for
historical reasons.

RAMP event submissions
======================

The above scores are also calculated when you make a submission to a RAMP
event. However, only the mean cv validation score (i.e.,
``valid  0.825 ± 0.0096`` above) is shown on the public leaderboard. The
mean cv test score is not shown as we wish to assess if the participants
submissions generalise to the private test data. Providing them with the
test score provides participants with a score to try and improve and may result
in models that perform well on the test data because it is overfit for the test
data.

Typically, the test score is used to officially rank the participants and
are made public at the end of a RAMP event.