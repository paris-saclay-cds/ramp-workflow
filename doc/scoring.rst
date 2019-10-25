.. _scoring:

RAMP scoring
############

Local submissions
=================

When testing a submission locally (i.e. with ``ramp_test_submission``) a number
of scores will be calculated and printed to standard output. The scores will
look like this::

    Testing Number of air passengers prediction
    Reading train and test files from ./data ...
    Reading cv ...
    Training ./submissions/starting_kit ...
    CV fold 0
        score   rmse
        train  0.324
        valid  0.819
        test   0.860
    CV fold 1
        score   rmse
        train  0.318
        valid  0.833
        test   0.845
    CV fold 2
        score   rmse
        train  0.330
        valid  0.827
        test   0.860
    CV fold 3
        score   rmse
        train  0.322
        valid  0.841
        test   0.863
    CV fold 4
        score   rmse
        train  0.326
        valid  0.831
        test   0.866
    CV fold 5
        score   rmse
        train  0.327
        valid  0.822
        test   0.874
    CV fold 6
        score   rmse
        train  0.327
        valid  0.814
        test   0.859
    CV fold 7
        score   rmse
        train  0.329
        valid  0.810
        test   0.861
    ----------------------------
    Mean CV scores
    ----------------------------
        score            rmse
        train  0.325 ± 0.0034
        valid  0.825 ± 0.0096
        test   0.861 ± 0.0076
    ----------------------------
    Bagged scores
    ----------------------------
        score   rmse
        valid  0.792
        test   0.825

Locally, there should be a training dataset and a testing dataset, usually
within a folder named ``data/``. We will call these datasets the 'public'
training data and the 'public' test data, as for a RAMP event, there will also
be private versions of these datasets.

Eight-fold cross-validation is performed whereby the public training data is
split into 'training' and 'validation' subsets 8 times. For each
cross-validation fold, the model is trained with the training data and used to
predict targets for the validation subset and the public testing data. The
scores are computed for the training, validation and testing datasets for each
fold. The mean of these 8 scores are calculated and printed under
``Mean CV scores``. 

``Bagged scores`` are calculated by combining the predictions of the 8 folds
and using the combined prediction to calculate the score. For regression
problems the combined prediction is just the mean of the predictions and
for classification problems, it is the mean probability of each class.
For example, the air passengers challenge aims to predict the number of
passengers on each flight. For each fold, different passenger number
predictions are made for the test data (as each model is differs because it
was trained using different data). The average of the 8 predictions from each
fold is calculated for each sample in the test data, and this new combined
prediction is used to calculate the 'bagged' score. This differs slightly for
the validation score because the validation datasets will be different
between each fold, and samples may or may not overlap between folds. In cases
where there is only one prediction for a validation sample, the combined
prediction will simply be the one prediction.

Note that technically this is not what bagging means, but the name used for
historical reasons.

RAMP event submissions
======================

The above scores are also calculated when you make a submission to a RAMP
event. However, only the mean cv validation score (i.e.,
``valid  0.825 ± 0.0096`` above) is shown on the public leaderboard. The
mean cv test score is not shown as we wish to test if the participants
submissions generalise to the private test data. Providing them with the
test score may result in models that perform well on the test data because
it is overfit for the test data.

Generally, the test score is used to officially rank the participants and
are made public at the end of a RAMP event.