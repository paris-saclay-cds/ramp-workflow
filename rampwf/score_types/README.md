# RAMP score types (metrics)

`score_types` is an abstraction of predictive quality metrics. Classes are wrappers of (typically scikit-learn) score metrics. Each score type has the following class variables:
* `is_lower_the_better` : boolean
* `minimum` : float
* `maximum` : float
* `worse` : float

  It is implemented in `BaseScoreType` as a property, used tytpically in loops to find the best of a list of scores.

Each score type has the following member variables:
* `name` : string

  It has a typical default that the user can redefine. Its main use is to name the columns of the leaderboard.
* `precision` : integer

  The number of digits after the decimal point when scores are displayed, for example, on the leaderboard.

Each score type has the following member functions:
* `score_function`

  It implements a wrapper function that takes
  * `ground_truths` : rampwf.prediction_types.BasePrediction
    The ground truth, typically a numpy array `y_true` wrapped into a prediction type.
  * `predictions` : rampwf.prediction_types.BasePrediction
    The prediction, typically a numpy array `y_pred` wrapped into a prediction type.

  It is called in `ramp-test` and at the [ramp-board][rboard] frontend when evaluating user submissions. Most of the time it uses the default implementation in `BaseScoreType` that checks if the number of rows are identical in `y_true` and `y_pred`, then calls `__call__(self, y_true, y_pred)`. `ClassifierBaseScoreType` does the same except it calls `__call__` with `y_true_label_index` and `y_pred_label_index`. When __call__ raises a non implemented error, `score_function` has to be overridden.
* `__call__`

  It implements a wrapper function that takes
  * `y_true` : np.array
    The ground truth.
  * `y_pred` : np.array
    The prediction.

  It is an optional function (when not implemented, e.g., in `Combined`, it raises an error) called by `score_function` or, typically, in user notebooks where folds are selected by the user and checks are unnecessary. It typically calls a scikit learn metrics or implements a metrics which doesn't exist there.

<!-- RAMP studio -->
[rstudio]: http://www.ramp.studio "RAMP main website"
[email]: mailto:admin@ramp.studio "Mailto: admin@ramp.studio"
[signup]: http://www.ramp.studio/sign-up "RAMP sign-up page"
[problems]: http://www.ramp.studio/problems "List of past RAMP challenges"
[themes]: http://www.ramp.studio/data_science_themes "Data science themes"
[domains]: http://www.ramp.studio/data_domains "Data domains"

<!-- git repos -->
[rworkflow]: https://github.com/paris-saclay-cds/ramp-workflow "Define RAMP score, workflow and CV scheme"
[rboard]: https://github.com/paris-saclay-cds/ramp-board "RAMP frontend library"
[rbackend]: https://github.com/paris-saclay-cds/ramp-backend "RAMP backend library (not implemented)"
[rdata]: https://github.com/ramp-data "Organization for RAMP open data sets"
[rkits]: https://github.com/ramp-kits "Organization for RAMP starting kits"
