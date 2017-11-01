"""The :mod:`rampwf.utils.command_line` submodule provide command line
utils.
"""
from __future__ import print_function

from os import listdir
from os.path import join, isdir

from .testing import assert_submission, assert_notebook, convert_notebook


def create_ramp_test_submission_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='ramp_test_submission',
        description='Test your ramp-kit before attempting a submission.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ramp_kit_dir',
                        default='.',
                        type=str,
                        help='Root directory of the ramp-kit to test.')
    parser.add_argument('--ramp_data_dir',
                        default='.',
                        type=str,
                        help='Directory containing the data. This directory'
                        ' should contain a "data" folder.')
    parser.add_argument('--submission',
                        default='starting_kit',
                        type=str,
                        help='The kit to test. It should be located in the'
                        ' "submissions" folder of the starting kit. If "ALL",'
                        ' all submissions in the directory will be tested.')
    parser.add_argument('--quick-test', dest='quick_test', action='store_true',
                        help='Specify this flag to test the submission on a '
                             'small subset of the data.'
                        )
    parser.add_argument('--pickle', dest='pickle', action='store_true',
                        help='Specify this flag to pickle the submission '
                             'after training.')
    parser.add_argument('--save-y-preds', dest='save_y_preds',
                        action='store_true',
                        help='Specify this flag to save preditions '
                             'after training.')
    parser.add_argument('--retrain', dest='retrain',
                        action='store_true',
                        help='Specify this flag to retrain the submission '
                             'on the full training set after the CV loop.')
    return parser


def ramp_test_submission():
    parser = create_ramp_test_submission_parser()
    args = parser.parse_args()

    if args.quick_test:
        import os
        os.environ['RAMP_TEST_MODE'] = '1'

    is_pickle = False
    if args.pickle:
        is_pickle = True

    save_y_preds = False
    if args.save_y_preds:
        save_y_preds = True

    retrain = False
    if args.retrain:
        retrain = True

    if args.submission == "ALL":
        ramp_submission_dir = join(args.ramp_kit_dir, 'submissions')
        submission = [directory
                      for directory in listdir(ramp_submission_dir)
                      if isdir(join(ramp_submission_dir, directory))]
    else:
        submission = [args.submission]

    for sub in submission:
        assert_submission(ramp_kit_dir=args.ramp_kit_dir,
                          ramp_data_dir=args.ramp_data_dir,
                          submission=sub,
                          is_pickle=is_pickle,
                          save_y_preds=save_y_preds,
                          retrain=retrain)


def create_ramp_test_notebook_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='ramp_test_notebook',
        description='Test your notebook before submitting a ramp.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ramp_kit_dir',
                        default='.',
                        type=str,
                        help='Root directory of the ramp-kit to test.')
    return parser


def ramp_test_notebook():
    parser = create_ramp_test_notebook_parser()
    args = parser.parse_args()
    assert_notebook(ramp_kit_dir=args.ramp_kit_dir)


def ramp_convert_notebook():
    parser = create_ramp_test_notebook_parser()
    args = parser.parse_args()
    convert_notebook(ramp_kit_dir=args.ramp_kit_dir)
