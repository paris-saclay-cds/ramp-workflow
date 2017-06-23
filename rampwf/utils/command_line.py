"""The :mod:`rampwf.utils.command_line` submodule provide command line
utils.
"""
from __future__ import print_function

from os import listdir
from os.path import join, isdir

from .testing import assert_submission


def create_parser():
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
    return parser


def ramp_test_submission():
    parser = create_parser()
    args = parser.parse_args()

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
                          submission=sub)
