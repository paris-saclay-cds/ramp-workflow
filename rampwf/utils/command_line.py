"""The :mod:`rampwf.utils.command_line` submodule provide command line
utils.
"""
from __future__ import print_function

from .testing import assert_submission


def ramp_test_submission():
    import argparse
    parser = argparse.ArgumentParser(
        prog='ramp_test_submission',
        description='Test your ramp-kit before attempting a submission.')
    parser.add_argument('--ramp_kit_dir',
                        default='.',
                        nargs=1,
                        type=str,
                        help='Root directory of the ramp-kit to test.')
    parser.add_argument('--ramp_data_dir',
                        default='.',
                        nargs=1,
                        type=str,
                        help='Directory containing the data. This directory'
                        ' should contain a "data" folder.')
    parser.add_argument('--submission_name',
                        default='starting_kit',
                        nargs=1,
                        type=str,
                        help='The kit to test. It should be located in the'
                        ' "submissions" folder of the starting kit.')

    args = parser.parse_args()
    assert_submission(ramp_kit_dir=args.ramp_kit_dir,
                      ramp_data_dir=args.ramp_data_dir,
                      submission_name=args.submission_name)
