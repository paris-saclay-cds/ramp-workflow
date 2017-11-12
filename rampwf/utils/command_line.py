"""The :mod:`rampwf.utils.command_line` submodule provide command line
utils.
"""
from __future__ import print_function

from os import listdir
from os.path import join, isdir

from .testing import (
    assert_submission, assert_notebook, convert_notebook, blend_submissions)


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
                        help='Specify this flag to save predictions '
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


def create_ramp_blend_submissions_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog='ramp_blend_submissions',
        description='Blend several submissions.',
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
    parser.add_argument('--submissions',
                        default='ALL',
                        type=str,
                        help='The submissions to blend. They should be located'
                        ' in the "submissions" folder of the starting kit. If'
                        ' "ALL", all submissions in the directory will be'
                        ' blended.')
    parser.add_argument('--save-y-preds', dest='save_y_preds',
                        action='store_true',
                        help='Specify this flag to save predictions '
                             'after training.')
    parser.add_argument('--min-improvement', dest='min_improvement',
                        default='0.0',
                        help='The minimum score improvement when adding.'
                        ' submissions to the ensemble.')
    return parser


def ramp_blend_submissions():
    parser = create_ramp_blend_submissions_parser()
    args = parser.parse_args()

    save_y_preds = False
    if args.save_y_preds:
        save_y_preds = True

    if args.submissions == "ALL":
        ramp_submission_dir = join(args.ramp_kit_dir, 'submissions')
        submissions = [directory
                       for directory in listdir(ramp_submission_dir)
                       if isdir(join(ramp_submission_dir, directory))]
    else:
        submissions = args.submissions.split(',')

    blend_submissions(ramp_kit_dir=args.ramp_kit_dir,
                      ramp_data_dir=args.ramp_data_dir,
                      submissions=submissions,
                      save_y_preds=save_y_preds,
                      min_improvement=float(args.min_improvement))
