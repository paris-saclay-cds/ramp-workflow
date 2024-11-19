# coding: utf-8

"""Provide utils to test ramp-kits."""

import sys
import time
import shutil
import itertools
from pathlib import Path
from functools import cache
from collections.abc import Iterable
import numpy as np
import pandas as pd
import rampwf as rw

from .combine import blend_on_fold, get_score_cv_bags
from .io import load_y_pred, load_predictions
from .importing import import_module_from_source
from .pretty_print import print_title, print_df_scores
from .notebook import execute_notebook, convert_notebook
from .scoring import round_df_scores, mean_score_matrix, reorder_df_scores
from .submission import run_submission_on_cv_fold, save_submissions
from .submission import run_submission_on_full_train


def assert_notebook(ramp_kit_dir="."):
    print("----------------------------")
    convert_notebook(ramp_kit_dir)
    execute_notebook(ramp_kit_dir)


def assert_read_problem(ramp_kit_dir="."):
    ramp_kit_dir = Path(ramp_kit_dir)
    # allowing external imports from the external_imports folder if it exists
    ext_imp_dir = ramp_kit_dir / "external_imports"
    if ext_imp_dir.exists() and str(ext_imp_dir) not in sys.path:
        sys.path.append(str(ext_imp_dir))
    # giving a random name to the module so it passes looped tests
    return import_module_from_source(ramp_kit_dir / "problem.py", "problem")


def assert_title(ramp_kit_dir="."):
    problem = assert_read_problem(ramp_kit_dir)
    print_title(f"Testing {problem.problem_title}")


@cache
def assert_data(ramp_kit_dir=".", ramp_data_dir=".", data_label=None):
    problem = assert_read_problem(ramp_kit_dir)
    data_label_dir = f"{data_label}/" if data_label is not None else ""
    print_title(
        f"Reading train and test files from {ramp_data_dir}/data/{data_label_dir}"
    )
    kwargs = {}
    if data_label is not None:
        kwargs["data_label"] = data_label
    X_train, y_train = problem.get_train_data(path=ramp_data_dir, **kwargs)
    X_test, y_test = problem.get_test_data(path=ramp_data_dir, **kwargs)
    return X_train, y_train, X_test, y_test


@cache
def _get_cv(
    ramp_kit_dir,
    ramp_data_dir,
    data_label,
    fold_idxs_tuple,
):
    # lists cannot be cached
    problem = assert_read_problem(ramp_kit_dir)
    X_train, y_train, _, _ = assert_data(
        ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir, data_label=data_label
    )
    print_title("Reading cv ...")
    if fold_idxs_tuple is None:
        cv = list(problem.get_cv(X_train, y_train))
    else:
        try:
            cv = list(problem.get_cv(X_train, y_train, fold_idxs_tuple))
        except TypeError:
            # get_cv does not accept fold_idxs
            cv = list(problem.get_cv(X_train, y_train))
            cv = [fold for fold_i, fold in enumerate(cv) if fold_i in fold_idxs_tuple]
    return cv


def assert_cv(
    ramp_kit_dir=".",
    ramp_data_dir=".",
    data_label=None,
    fold_idxs=None,
):
    if fold_idxs is None:
        return _get_cv(
            ramp_kit_dir,
            ramp_data_dir,
            data_label,
            fold_idxs_tuple=None,
        )
    else:
        return _get_cv(
            ramp_kit_dir,
            ramp_data_dir,
            data_label,
            fold_idxs_tuple=tuple(fold_idxs),
        )


def assert_score_types(ramp_kit_dir="."):
    problem = assert_read_problem(ramp_kit_dir)
    score_types = problem.score_types
    return score_types


def assert_submission(
    ramp_kit_dir=".",
    ramp_data_dir=".",
    ramp_submission_dir=None,
    data_label=None,
    submission="starting_kit",
    is_pickle=False,
    is_partial_train=False,
    save_output=False,
    retrain=False,
    bag=True,
    force_retrain=True,
    fold_idxs=None,
):
    """Helper to test a submission from a ramp-kit.

    Parameters
    ----------
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be tested for submission.
    ramp_data_dir : str, default='.'
        The directory of the data.
    ramp_submission_dir : str, default='./submissions'
        The directory of the submissions.
    data_label : str, default=None
        The subdirectory of data in /data and training outputs in
        /submissions/<submission>/training_output
    submission : str, default='starting_kit'
        The name of the submission to be tested.
    is_pickle : bool, default is False
        Whether to pickle the trained workflow or not.
    is_partial_train : bool, default is False
        Whether to partial train a trained workflow, pickled before.
        workflow.train_submission needs to accept prev_trained_workflow.
    save_y_preds : bool, default is False
        Whether to store the predictions.
    retrain : bool, default is False
        Whether to train the workflow on the full training set and
        test on the test set.
    bag : bool, default is True
        Whether to bag the submission.
    force_retrain : bool, default is True
        Whether to retrain the folds which have scores.csv.
    fold_idxs : list of int, default=None
        Fold indices to train on.
        If None, we will train on all folds.
    """
    if ramp_submission_dir is None:
        ramp_submission_dir = Path(ramp_kit_dir) / "submissions"
    ramp_submission_dir = Path(ramp_submission_dir)
    submission_path = ramp_submission_dir / submission
    print_title(f"Training {submission_path} ...")

    problem = assert_read_problem(ramp_kit_dir)
    assert_title(ramp_kit_dir)
    X_train, y_train, X_test, y_test = assert_data(
        ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir, data_label=data_label
    )
    cv = assert_cv(ramp_kit_dir, ramp_data_dir, data_label, fold_idxs)
    workflow = problem.workflow
    try:
        workflow.metadata = problem.get_metadata(ramp_data_dir, data_label)
    except AttributeError:
        print("No metadata")
    workflow.set_element_names(submission_path)
    score_types = assert_score_types(ramp_kit_dir)
    print_title("Preprocessing data")
    t0 = time.time()
    X_train, y_train, X_test = workflow.preprocess_data(
        submission_path, X_train, y_train, X_test
    )
    preprocessing_time = time.time() - t0
    print_title(f"Preprocessing time: {preprocessing_time}")

    training_output_path = ""
    if is_pickle or save_output:
        # creating <submission_path>/<submission>/training_output dir
        # optionally
        # <submission_path>/<submission>/training_output/<data_label>
        training_output_path = submission_path / "training_output"
        if data_label is not None:
            training_output_path = training_output_path / data_label
        training_output_path.mkdir(parents=True, exist_ok=True)
        print(f"Training output path: {training_output_path}")

    # saving predictions for CV bagging after the CV loop
    predictions_valid_list = []
    predictions_test_list = []
    df_scores_list = []

    # if no folds given, we may just want to retrain on full training fold
    if fold_idxs is None or len(fold_idxs) > 0:
        if fold_idxs is None:
            fold_gen = enumerate(cv)
        else:
            fold_gen = zip(fold_idxs, cv)
        for fold_i, fold in fold_gen:
            fold_output_path = ""
            if is_pickle or save_output:
                # creating <submission_path>/<submission>/training_output/fold_<i>
                fold_output_path = training_output_path / f"fold_{fold_i}"
                fold_output_path.mkdir(parents=True, exist_ok=True)
            print_title(f"CV fold {fold_i}")

            do_train = True
            if not force_retrain:
                do_train = False
                try:
                    df_scores = pd.read_csv(fold_output_path / "scores.csv")
                    df_scores = df_scores.set_index("step")
                    predictions_valid, predictions_test = load_predictions(
                        problem,
                        fold[1],
                        data_path=ramp_data_dir,
                        input_path=fold_output_path,
                    )
                    print("Scores and predictions found and loaded, not retraining")
                except:
                    do_train = True
            if do_train:
                predictions_valid, predictions_test, df_scores = (
                    run_submission_on_cv_fold(
                        problem,
                        submission_path,
                        fold,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        is_pickle,
                        is_partial_train,
                        save_output,
                        fold_output_path,
                        ramp_data_dir,
                    )
                )

            # saving predictions for CV bagging after the CV loop
            df_scores_list.append(df_scores)
            predictions_valid_list.append(predictions_valid)
            predictions_test_list.append(predictions_test)

        print_title("----------------------------")
        print_title("Mean CV scores")
        print_title("----------------------------")
        df_mean_scores = mean_score_matrix(df_scores_list, score_types)
        print_df_scores(df_mean_scores, indent="\t")

        if bag:
            bag_submissions(
                problem,
                X_train,
                y_train,
                y_test,
                predictions_valid_list,
                predictions_test_list,
                training_output_path,
                ramp_kit_dir=ramp_kit_dir,
                ramp_data_dir=ramp_data_dir,
                data_label=data_label,
                score_type_index=None,
                save_output=save_output,
                fold_idxs=fold_idxs,
            )

    if retrain:
        # We retrain on the full training set
        print_title("----------------------------")
        print_title("Retrain scores")
        print_title("----------------------------")
        run_submission_on_full_train(
            problem,
            submission_path,
            X_train,
            y_train,
            X_test,
            y_test,
            score_types,
            is_pickle,
            save_output,
            training_output_path,
            ramp_data_dir,
        )


def bag_submissions(
    problem,
    X_train,
    y_train,
    y_test,
    predictions_valid_list,
    predictions_test_list,
    training_output_path,
    ramp_kit_dir=".",
    ramp_data_dir=".",
    data_label=None,
    score_type_index=0,
    save_output=False,
    score_table_title="Bagged scores",
    score_f_name_prefix="",
    fold_idxs=None,
    bag_ranks=False,
    test=True,
):
    """CV-bag trained submission.

    Parameters
    ----------
    problem : problem object
        imp.loaded from problem.py
    y_train : a list of training ground truth
        returned by problem.get_train_data
    y_test : a list of testing ground truth or None
        returned by problem.get_test_data
    predictions_valid_list : list of Prediction objects
        returned by run_submission_on_cv_fold
    predictions_test_list : list of Prediction objects or None
        returned by run_submission_on_cv_fold
    training_output_path : Path
        submissions/<submission>/training_output
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be tested for submission.
    ramp_data_dir : str
        the directory of the data
    data_label : str, default=None
        The subdirectory of data in /data and training outputs in
        /submissions/<submission>/training_output
    score_type_index : int or None.
        The score type on which we bag. If None, all scores will be computed.
    save_output : boolean
        True if predictions should be written in files
    score_table_title : str
    score_f_name_prefix : str
    fold_idxs : list of int, default=None
        Fold indices to bag.
        If None, we will bag all folds.
    bag_ranks : boolean
        With some rank-based scores like auc or ngini, it is better to normalize
        the predictions to their rank, but only when bagginin the blends, otherwise
        uncalibrated models and non-homogeneous models per fold create irregular
        blended rankings.
    test : bool, default=True
        Whether to use test.
    """
    print_title("----------------------------")
    print_title(score_table_title)
    print_title("----------------------------")
    score_type_index = slice(None) if score_type_index is None else score_type_index
    score_types = problem.score_types[score_type_index]
    score_types = (
        [score_types] if not isinstance(score_types, Iterable) else score_types
    )
    cv = assert_cv(ramp_kit_dir, ramp_data_dir, data_label, fold_idxs)
    if fold_idxs is None:
        fold_idxs = list(range(len(cv)))

    # With some rank-based scores like auc or ngini, it is better to normalize
    # the predictions to their rank, but only when bagginin the blends, otherwise
    # uncalibrated models and non-homogeneous models per fold create irregular
    # blended rankings.
    if test:
        prediction_lists = [predictions_valid_list, predictions_test_list]
    else:
        prediction_lists = [predictions_valid_list]
    if bag_ranks:
        for prediction_list in prediction_lists:
            for prediction in prediction_list:
                try:
                    prediction.rankify()
                except AttributeError:
                    pass

    # placeholder to store the scores and predictions
    bagged_scores = {}
    if test:
        valid_and_test = y_test is not None
    else:
        valid_and_test = False
    scoring_step = ["valid", "test"] if valid_and_test else ["valid"]
    for step in scoring_step:
        if step == "valid":
            test_idx = [fold[1] for fold in cv]
            pred_list = predictions_valid_list
            y_step = y_train
        else:
            test_idx = None
            pred_list = predictions_test_list
            y_step = y_test
        gt_list = problem.Predictions(y_true=y_step)
        pred, score_dict = get_score_cv_bags(
            score_types, pred_list, gt_list, test_is_list=test_idx
        )
        bagged_scores[step] = score_dict
        # the predictions will always be the same for all score and we store
        # only a single instance
        if save_output:
            save_submissions(
                problem,
                pred.y_pred,
                data_path=ramp_data_dir,
                output_path=training_output_path,
                suffix=f"{score_f_name_prefix}bagged_{step}",
            )

    df_scores = pd.concat(
        {step: pd.DataFrame(scores) for step, scores in bagged_scores.items()}
    )
    df_scores.columns = df_scores.columns.rename("score")
    df_scores.index = df_scores.index.rename(["step", "n_bag"])
    if y_test is None or not test:
        df_scores["fold_idx"] = list(fold_idxs)  # valid
    else:
        df_scores["fold_idx"] = list(fold_idxs) + list(fold_idxs)
    # bagging learning curves can be plotted on this df_scores
    if save_output:
        df_scores.to_csv(training_output_path / "bagged_scores.csv")

    # prepare the bagged scores which will be printed.
    highest_level = df_scores.index.get_level_values("n_bag").max()
    df_scores = df_scores.loc[(slice(None), highest_level), :]
    df_scores.index = df_scores.index.droplevel("n_bag")
    df_scores = reorder_df_scores(df_scores, score_types)
    df_scores = round_df_scores(df_scores, score_types)
    print_df_scores(df_scores, indent="\t")


def blend_submissions(
    submissions,
    ramp_kit_dir=".",
    ramp_data_dir=".",
    ramp_submission_dir=None,
    data_label=None,
    save_output=False,
    min_improvement=0.0,
    score_type_index=0,
    output_path=None,
    fold_idxs=None,
    bag_ranks=False,
    test=True,
):
    """Blending submissions in a ramp-kit and compute contributivities.

    First blend submissions on each fold, then bag the blended submissions.
    Generates graded contributivities, more noise than bag_then_blend.
    Does not work well with some scores where models may have different
    calibratedness (like auc or ngini which only depend on order).

    If save_output is True, we create three files:
    <ramp_submission_dir>/training_output/contributivities.csv
    <ramp_submission_dir>/training_output/bagged_scores_combined.csv
    <ramp_submission_dir>/training_output/bagged_scores_foldwise_best.csv

    Parameters
    ----------
    submissions : list of str
        List of submission names (folders in <ramp_submission_dir>).
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be blended.
    ramp_data_dir : str, default='.'
        The directory of the data.
    data_label : str, default=None
        The subdirectory of data in {ramp_data_dir}/data and training outputs
        in {ramp_kit_dir}/submissions/<submission>/training_output
    ramp_submission_dir : str, default='submissions'
        The directory of the submissions relative to {ramp_kit_dir}.
    save_output : bool, default is False
        Whether to store the blending results.
    min_improvement : float, default is 0.0
        The minimum improvement under which greedy blender is stopped.
    output_path : str, default is None
        The folder where the blended scores and controbutivities are saved.
        If None, it is <ramp_submission_dir>/[<data_label>]/training_output
    fold_idxs : list of int, default=None
        Fold indices to blend.
        If None, we will blend all folds.
    bag_ranks : boolean
        With some rank-based scores like auc or ngini, it is better to normalize
        the predictions to their rank, but only when bagging the blends, otherwise
        uncalibrated models and non-homogeneous models per fold create irregular
        blended rankings.
    test : bool, default=True
        Whether to use test.
    """
    if ramp_submission_dir is None:
        ramp_submission_dir = Path(ramp_kit_dir) / "submissions"
    ramp_submission_dir = Path(ramp_submission_dir)
    problem = assert_read_problem(ramp_kit_dir)
    print_title(f"Blending {problem.problem_title}")
    X_train, y_train, X_test, y_test = assert_data(
        ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir, data_label=data_label
    )
    cv = assert_cv(ramp_kit_dir, ramp_data_dir, data_label, fold_idxs)
    if fold_idxs is None:
        fold_idxs = list(range(len(cv)))
    score_types = assert_score_types(ramp_kit_dir)
    # collect real folds where there are >0 models that can be blended
    n_real_folds = 0
    real_fold_idxs = []
    contributivitys = np.zeros((len(submissions), len(cv)))

    combined_predictions_valid_list = []
    foldwise_best_predictions_valid_list = []
    combined_predictions_test_list = []
    foldwise_best_predictions_test_list = []
    for fold_i, fold in zip(fold_idxs, cv):
        print_title(f"CV fold {fold_i}")
        ground_truths_valid = problem.Predictions(y_true=y_train, fold_is=fold[1])
        predictions_valid_list = []
        predictions_test_list = []
        submission_is = []
        for submission_i, submission in enumerate(submissions):
            submission_path = ramp_submission_dir / submission
            training_output_path = submission_path / "training_output"
            if data_label is not None:
                training_output_path = training_output_path / data_label
            fold_output_path = training_output_path / f"fold_{fold_i}"
            try:
                predictions_valid, predictions_test = load_predictions(
                    problem,
                    fold[1],
                    data_path=ramp_data_dir,
                    input_path=fold_output_path,
                    test=test,
                )
                predictions_valid_list.append(predictions_valid)
                predictions_test_list.append(predictions_test)
                submission_is.append(submission_i)
            except:
                pass
        # Only bag fold if there is at least one submission trained on it
        if len(predictions_valid_list) > 0:
            best_index_list = blend_on_fold(
                predictions_valid_list,
                ground_truths_valid,
                score_types[score_type_index],
                min_improvement=min_improvement,
            )

            # we share a unit of 1. among the contributive submissions
            unit_contributivity = 1.0 / len(best_index_list)
            for best_i in best_index_list:
                contributivitys[submission_is[best_i], n_real_folds] += (
                    unit_contributivity
                )

            combined_predictions_valid_list.append(
                problem.Predictions.combine(predictions_valid_list, best_index_list)
            )
            foldwise_best_predictions_valid_list.append(
                predictions_valid_list[best_index_list[0]]
            )
            if test:
                combined_predictions_test_list.append(
                    problem.Predictions.combine(predictions_test_list, best_index_list)
                )
                foldwise_best_predictions_test_list.append(
                    predictions_test_list[best_index_list[0]]
                )

            n_real_folds += 1
            real_fold_idxs.append(fold_i)

    if n_real_folds == 0:
        print("No folds to blend")
        return

    contributivitys /= n_real_folds
    contributivitys_df = pd.DataFrame()
    contributivitys_df["submission"] = np.array(submissions)
    contributivitys_df["contributivity"] = np.zeros(len(submissions))
    for i, fold_i in enumerate(real_fold_idxs):
        c_i = contributivitys[:, i]
        contributivitys_df["fold_{}".format(fold_i)] = c_i
        contributivitys_df["contributivity"] += c_i
    permilage_factor = 1000 / contributivitys_df["contributivity"].sum()
    contributivitys_df["contributivity"] *= permilage_factor
    rounded = contributivitys_df["contributivity"].round().astype(int)
    contributivitys_df["contributivity"] = rounded
    contributivitys_df = contributivitys_df.sort_values(
        "contributivity", ascending=False
    )
    print(contributivitys_df.to_string(index=False))

    if output_path is None:
        output_path = ramp_submission_dir / "training_output"
        if data_label is not None:
            output_path = output_path / data_label
    else:
        output_path = Path(output_path)
    if save_output:
        output_path.mkdir(parents=True, exist_ok=True)
        contributivitys_df.to_csv(output_path / "contributivities.csv", index=False)

    # bagging the foldwise ensembles
    bag_submissions(
        problem,
        X_train,
        y_train,
        y_test,
        combined_predictions_valid_list,
        combined_predictions_test_list,
        output_path,
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
        data_label=data_label,
        score_type_index=score_type_index,
        save_output=save_output,
        score_table_title="Combined bagged scores",
        score_f_name_prefix="combined_",
        fold_idxs=real_fold_idxs,
        bag_ranks=bag_ranks,
        test=test
    )
    if save_output:
        shutil.move(
            output_path / "bagged_scores.csv",
            output_path / "bagged_scores_combined.csv",
        )

    # bagging the foldwise best submissions
    bag_submissions(
        problem,
        X_train,
        y_train,
        y_test,
        foldwise_best_predictions_valid_list,
        foldwise_best_predictions_test_list,
        output_path,
        ramp_kit_dir=ramp_kit_dir,
        ramp_data_dir=ramp_data_dir,
        data_label=data_label,
        score_type_index=score_type_index,
        save_output=save_output,
        score_table_title="Foldwise best bagged scores",
        score_f_name_prefix="foldwise_best_",
        fold_idxs=real_fold_idxs,
        bag_ranks=bag_ranks,
        test=test,
    )
    if save_output:
        shutil.move(
            output_path / "bagged_scores.csv",
            output_path / "bagged_scores_foldwise_best.csv",
        )


def bag_then_blend_submissions(
    submissions,
    ramp_kit_dir=".",
    ramp_data_dir=".",
    ramp_submission_dir=None,
    data_label=None,
    save_output=False,
    min_improvement=0.0,
    score_type_index=0,
    output_path=None,
    fold_idxs=None,
    test=True
):
    """Blending submissions in a ramp-kit and compute contributivities.

    First bag the submissions on all the folds, then blend the bagged submissions.
    Generates smaller ensembles than blend.

    If save_output is True, we create three files:
    <ramp_submission_dir>/training_output/contributivities.csv
    <ramp_submission_dir>/training_output/bagged_scores_combined.csv
    <ramp_submission_dir>/training_output/bagged_scores_foldwise_best.csv

    Parameters
    ----------
    submissions : list of str
        List of submission names (folders in <ramp_submission_dir>).
    ramp_kit_dir : str, default='.'
        The directory of the ramp-kit to be blended.
    ramp_data_dir : str, default='.'
        The directory of the data.
    data_label : str, default=None
        The subdirectory of data in {ramp_data_dir}/data and training outputs
        in {ramp_kit_dir}/submissions/<submission>/training_output
    ramp_submission_dir : str, default='submissions'
        The directory of the submissions relative to {ramp_kit_dir}.
    save_output : bool, default is False
        Whether to store the blending results.
    min_improvement : float, default is 0.0
        The minimum improvement under which greedy blender is stopped.
    output_path : str, default is None
        The folder where the blended scores and controbutivities are saved.
        If None, it is <ramp_submission_dir>/[<data_label>]/training_output
    fold_idxs : list of int, default=None
        Fold indices to blend.
        If None, we will blend all folds.
    test : bool, default=True
        Whether to use test.
    """
    if ramp_submission_dir is None:
        ramp_submission_dir = Path(ramp_kit_dir) / "submissions"
    ramp_submission_dir = Path(ramp_submission_dir)
    problem = assert_read_problem(ramp_kit_dir)
    print_title(f"Bagging then blending {problem.problem_title}")
    X_train, y_train, X_test, y_test = assert_data(
        ramp_kit_dir, ramp_data_dir, data_label
    )
    cv = assert_cv(ramp_kit_dir, ramp_data_dir, data_label, fold_idxs)
    if fold_idxs is None:
        fold_idxs = list(range(len(cv)))
    score_types = assert_score_types(ramp_kit_dir)

    bagged_predictions_valid_list = []
    bagged_predictions_test_list = []
    submission_is = []
    for submission_i, submission in enumerate(submissions):
        print(submission)
        submission_path = ramp_submission_dir / submission
        training_output_path = submission_path / "training_output"
        if data_label is not None:
            training_output_path = training_output_path / data_label
        predictions_valid_list = []
        predictions_test_list = []
        for fold_i, fold in zip(fold_idxs, cv):
            fold_output_path = training_output_path / f"fold_{fold_i}"
            predictions_valid, predictions_test = load_predictions(
                problem,
                fold[1],
                data_path=ramp_data_dir,
                input_path=fold_output_path,
                test=test
            )
            predictions_valid_list.append(predictions_valid)
            predictions_test_list.append(predictions_test)
        submission_is.append(submission_i)

        if test:
            valid_and_test = y_test is not None
        else:
            valid_and_test = False
        scoring_step = ["valid", "test"] if valid_and_test else ["valid"]
        for step in scoring_step:
            if step == "valid":
                test_idx = [fold[1] for fold in cv]
                pred_list = predictions_valid_list
                y_step = y_train
                predictions_list = bagged_predictions_valid_list
            else:
                test_idx = None
                pred_list = predictions_test_list
                y_step = y_test
                predictions_list = bagged_predictions_test_list
            gt_list = problem.Predictions(y_true=y_step)
            predictions, score_dict = get_score_cv_bags(
                score_types, pred_list, gt_list, test_is_list=test_idx
            )
            predictions_list.append(predictions)

    valid_overlap_is = []
    for fold in cv:
        valid_overlap_is += list(fold[1])
    valid_overlap_is = np.unique(valid_overlap_is)

    if len(bagged_predictions_valid_list) > 0:
        best_index_list = blend_on_fold(
            predictions_list = bagged_predictions_valid_list,
            ground_truths_valid = problem.Predictions(y_true=y_train, fold_is=valid_overlap_is),
            score_type = score_types[0],
            min_improvement = min_improvement,
        )

        contributivitys = np.zeros(len(submissions))
        # we share a unit of 1. among the contributive submissions
        unit_contributivity = 1.0 / len(best_index_list)
        for best_i in best_index_list:
            contributivitys[submission_is[best_i]] += unit_contributivity

        contributivitys_df = pd.DataFrame()
        contributivitys_df["submission"] = np.array(submissions)
        contributivitys_df["contributivity"] = contributivitys
        permilage_factor = 1000 / contributivitys_df["contributivity"].sum()
        contributivitys_df["contributivity"] *= permilage_factor
        rounded = contributivitys_df["contributivity"].round().astype(int)
        contributivitys_df["contributivity"] = rounded
        contributivitys_df = contributivitys_df.sort_values(
            "contributivity", ascending=False
        )
        print(contributivitys_df.to_string(index=False))

        combined_predictions_valid = problem.Predictions.combine(
            bagged_predictions_valid_list, best_index_list)
        if test:
            combined_predictions_test = problem.Predictions.combine(
                bagged_predictions_test_list, best_index_list)
        if output_path is None:
            output_path = ramp_submission_dir / "training_output"
            if data_label is not None:
                output_path = output_path / data_label
        else:
            output_path = Path(output_path)
        if save_output:
            df = pd.DataFrame()
            for step in scoring_step:
                if step == "valid":
                    fold_is = valid_overlap_is
                    pred = combined_predictions_valid
                    y_step = y_train
                else:
                    fold_is = None
                    pred = combined_predictions_test
                    y_step = y_test
                gt = problem.Predictions(y_true=y_step, fold_is=fold_is)
                save_submissions(
                    problem,
                    pred.y_pred,
                    data_path=ramp_data_dir,
                    output_path=output_path,
                    suffix=f"bagged_then_blended_{step}",
                )
                score = score_types[0].score_function(gt, pred)
                df[step] = [score]
            df.to_csv(output_path / "bagged_then_blended_scores.csv", index=False)
            contributivitys_df.to_csv(output_path / "contributivities_bagged_then_blended.csv", index=False)

