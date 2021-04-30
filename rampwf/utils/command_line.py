# -*- coding: utf-8 -*-
"""The :mod:`rampwf.utils.command_line` submodule provide command line
utils.
"""
from __future__ import print_function
from __future__ import unicode_literals
import re
import sys
import os
from os import listdir
from os.path import join, isdir
from collections import defaultdict
import numpy as np
import pandas as pd

from .testing import blend_submissions


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
    parser.add_argument('--ramp_submission_dir',
                        default='submissions',
                        type=str,
                        help='Directory where the submissions are stored. It '
                        'should contain a "submissions" directory.')
    parser.add_argument('--submissions',
                        default='ALL',
                        type=str,
                        help='The submissions to blend. They should be located'
                        ' in the "submissions" folder of the starting kit.'
                        ' Specify submissions separated by a comma without'
                        ' spaces. If "ALL", all submissions in the directory'
                        ' will be blended.')
    parser.add_argument('--save-output', dest='save_output',
                        action='store_true',
                        help='Specify this flag to save predictions '
                             'after blending.')
    parser.add_argument('--min-improvement', dest='min_improvement',
                        default='0.0',
                        help='The minimum score improvement when adding.'
                        ' submissions to the ensemble.')
    return parser


def ramp_blend_submissions():
    parser = create_ramp_blend_submissions_parser()
    args = parser.parse_args()

    save_output = False
    if args.save_output:
        save_output = True

    if args.submissions == "ALL":
        ramp_submission_dir = join(args.ramp_kit_dir, 'submissions')
        submissions = [directory
                       for directory in listdir(ramp_submission_dir)
                       if isdir(join(ramp_submission_dir, directory))]
    else:
        submissions = args.submissions.split(',')

    blend_submissions(ramp_kit_dir=args.ramp_kit_dir,
                      ramp_data_dir=args.ramp_data_dir,
                      ramp_submission_dir=args.ramp_submission_dir,
                      submissions=submissions,
                      save_output=save_output,
                      min_improvement=float(args.min_improvement))
