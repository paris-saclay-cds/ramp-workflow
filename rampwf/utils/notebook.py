# coding: utf-8
"""
Utilities to deal with Jupyter notebooks
"""
from __future__ import print_function

import os
import subprocess


def delete_line_from_file(f_name, line_to_delete):
    with open(f_name, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if line != line_to_delete:
                f.write(line)
        f.truncate()


def execute_notebook(ramp_kit_dir='.'):
    problem_name = os.path.abspath(ramp_kit_dir).split('/')[-1]
    print('Testing if the notebook can be executed')
    subprocess.call(
        'jupyter nbconvert --execute {}/{}_starting_kit.ipynb'
        .format(ramp_kit_dir, problem_name)+
        '--ExecutePreprocessor.timeout=600', shell=True)


def convert_notebook(ramp_kit_dir='.'):
    problem_name = os.path.abspath(ramp_kit_dir).split('/')[-1]
    print('Testing if the notebook can be converted to html')
    subprocess.call(
        'jupyter nbconvert --to html {}/{}_starting_kit.ipynb'
        .format(ramp_kit_dir, problem_name), shell=True)
    delete_line_from_file(
        '{}/{}_starting_kit.html'.format(ramp_kit_dir, problem_name),
        '<link rel="stylesheet" href="custom.css">\n')
