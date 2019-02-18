import os
import re
import itertools
import random
import tempfile
import shutil
from .testing import assert_read_problem
from .submission import run_submission_on_cv_fold
from .scoring import round_df_scores

HYPERPARAMS_REPL_REGEX = re.compile('# RAMP START HYPERPARAMETERS.*# RAMP END',
                                    re.S)
HYPERPARAMS_REPL_PATTERN = '\n'.join('# RAMP START HYPERPARAMETERS',
                                     'hyper_parameters = {}',
                                     '# RAMP END HYPERPARAMETERS')


def _make_random_parameters_space(params_distribution, seed=None):
    """Make a random set of hyper parameters sampled from `params_distribution`.

    Parameters:
    -----------
    params_distribution: dict
       a dictionay whose keys are hyperparameters names and values are
       iterables of possible values for each parameter.
    seed:
       random state value used for shuffling. Default: None.

    Returns:
    --------
    params_space: generator
        generator of all random combinations of hyper parameters. Each element
        will be a dictionary of parameters names and values.
    """
    param_names, param_values = list(zip(*params_distribution.items()))
    space = list(itertools.product(*param_values))
    random.seed(seed)
    random.shuffle(space)
    for values in space:
        yield dict(zip(param_names, values))


def _configure_submission(params, problem, submission_path, outdir):
    """Make a new instance of a submission configured with `params`.
    This function replaces the hyperparameters bloc of a submission
    with the configuration given by `params`. submission code is
    expected to have a block of the form:

    # RAMP START HYPERPARAMETERS
    hyper_parameters = {...}
    # RAMP END HYPERPARAMETERS"

    Parameters:
    -----------
    params: dict
        a dictionary of parameters names and values
    problem: module
        instance of submission interface
    submission_path: str
        path to submission code
    outdir: str
        output directory where the new submission will be saved
    """
    filename = os.path.join(submission_path,
                            problem.workflow.element_names[1] + ".py")
    # update hyperparameters section of estimator module
    with open(filename) as fp:
        text = fp.read()
        repl = HYPERPARAMS_REPL_PATTERN.format(params)
        text = HYPERPARAMS_REPL_REGEX.sub(repl, text)
    with open(filename, "w") as fp:
        fp.write(text)


def _eval_submission_with_params(params, problem, submission_path,
                                 X_train, y_train):
    """Evaluate a submission with a set of hyper parameters.

    Parameters:
    -----------
    params: dict
        a dictionary of parameters names and values
    problem: module
        instance of submission interface
    submission_path: str
        path to submission code
    X_train: ndarray
        train data
    y_train: ndarray
        train targets
    """
    print("Run with hyperparams:", params)
    with tempfile.TemporaryDirectory() as dirname:
        submission_name = os.path.split(submission_path.rstrip(os.path.sep))[1]
        temp_submission_path = os.path.join(dirname, submission_name)
        shutil.copytree(submission_path, temp_submission_path)

        _configure_submission(params, problem, temp_submission_path, dirname)

        df_scores_list = []
        predictions_valid_list = []

        for fold_i, fold in enumerate(problem.get_cv(X_train, y_train)):

            predictions_valid, _, df_scores = \
                run_submission_on_cv_fold(
                    problem, temp_submission_path, X_train, y_train, None,
                    None, problem.score_types, False, False, None,
                    fold, None)

            df_scores_rounded = round_df_scores(df_scores, problem.score_types)

            # saving predictions for CV bagging after the CV loop
            df_scores_list.append(df_scores_rounded)
            predictions_valid_list.append(predictions_valid)

    return df_scores_list, predictions_valid_list


def tune_hyper_parameters(params_distribution, ramp_kit_dir, submission_path,
                          n_iters=10, n_jobs=1, seed=None):
    """
    Try up to `n_iter` sets of parameters configuration for submission and
    return scores and the respective sets of parameters parameters.

    Parameters:
    -----------
    params_distribution: dict ({"param1" : [...], ...})
        distribution of pramaters to use for optimization
    ramp_kit_dir: str
        path to ramp kit directory
    submission_path: srt
        path to submission directory (relative to ramp_kit_dir) whose
        hyperparamers are to be optimized.
    n_iter: int, default: None
        maximum number parameters configuration to tyr.
    n_job: int, default: 1
        number of parallel processus to run.
    seed: hashable object, default: None
        used to initialize random space generator's state

    Returns:
    --------
    scores_and_params: list
        list of tuples of Cross-validation scores and parameters.
    """
    problem = assert_read_problem(ramp_kit_dir)
    submission_path = os.path.join(ramp_kit_dir, submission_path)
    X_train, y_train = problem.get_train_data(path=ramp_kit_dir)
    params_space = _make_random_parameters_space(params_distribution,
                                                 seed=seed)
    results = []
    for i in range(n_iters):
        try:
            params = next(params_space)
        except StopIteration:
            break
        scores = _eval_submission_with_params(params, problem, submission_path,
                                              X_train, y_train)
        results.append((scores, params))
    return results
