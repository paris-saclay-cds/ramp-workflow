# coding: utf-8
"""
Utilities for saving and loading predictions
"""
import os

import numpy as np

from .pretty_print import print_warning


def save_y_pred(problem, y_pred, data_path='.', output_path='.',
                suffix='test'):
    """Save a prediction vector in file.

    If problem.save_y_pred is implemented, y_pred is passed to it. Otherwise,
    np.savez_compressed is used on y_pred. If it crashes, a warning is raised.
    The file is (typically) in
    submissions/<submission_name>/training_output/y_pred_<suffix>.npz or
    submissions/<submission_name>/training_output/fold_<i>/y_pred_<suffix>.npz.

    Parameters
    ----------
    problem : a problem object
        loaded from problem.py, may implement save_y_pred
    y_pred : a prediction vector
        a vector of predictions to be saved
    data_path : str, (default='.')
        the directory of the ramp-kit to be tested for submission, maybe
        needed by problem.save_y_pred for, e.g., merging with an index vector
    output_path : str, (default='.')
        the directory where (typically) y_pred_<suffix>.npz will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.npz, can be used in
        problem.save_y_pred to, e.g., save only test predictions
    """
    try:
        # We try using custom made problem.save_y_pred
        # obligatory to implement if np.savez_compressed doesn't work on y_pred
        problem.save_y_pred(y_pred, data_path, output_path, suffix)
    except AttributeError:
        # We fall back to numpy savez_compressed
        try:
            y_pred_f_name = os.path.join(output_path,
                                         'y_pred_{}'.format(suffix))
            np.savez_compressed(y_pred_f_name, y_pred=y_pred)
        except Exception as e:
            print_warning(
                "Warning: model can't be saved.\n{}\n".format(e) +
                'Consider implementing custom save_y_pred in problem.py\n')


def load_y_pred(problem, data_path='.', input_path='.', suffix='test'):
    """Load a file into a prediction vector.

    If problem.load_y_pred is implemented, y_pred is loaded by it. Otherwise,
    np.load is used. If it crashes, the exception is raised. The input file
    is (typically) in
    submissions/<submission_name>/training_output/y_pred_<suffix>.npz or
    submissions/<submission_name>/training_output/fold_<i>/y_pred_<suffix>.npz.

    Parameters
    ----------
    problem : a problem object
        loaded from problem.py, may implement save_y_pred
    data_path : str, (default='.')
        the directory of the ramp-kit to be tested for submission, maybe
        needed by problem.save_y_pred for, e.g., merging with an index vector
    input_path : str, (default='.')
        the directory where (typically) y_pred_<suffix>.npz will be saved
    suffix : str, (default='test')
        suffix in (typically) y_pred_<suffix>.npz
    """
    try:
        # We try using custom made problem.load_y_pred
        # obligatory to implement if np.load doesn't work
        return problem.load_y_pred(data_path, input_path, suffix)
    except AttributeError:
        # We fall back to numpy load
        y_pred_f_name = os.path.join(input_path,
                                     'y_pred_{}.npz'.format(suffix))
        return np.load(y_pred_f_name)['y_pred']
