# coding: utf-8
"""
Utilities to deal with Jupyter notebooks
"""
from __future__ import print_function

import os
import sys

import nbformat
import nbconvert
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor


def delete_line_from_file(f_name, line_to_delete):
    with open(f_name, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if line != line_to_delete:
                f.write(line)
        f.truncate()


def execute_notebook(ramp_kit_dir='.'):
    problem_name = os.path.basename(os.path.abspath(ramp_kit_dir))
    print('Testing if the notebook can be executed')
    notebook_filename = os.path.join(
        os.path.abspath(ramp_kit_dir),
        '{}_starting_kit.ipynb'.format(problem_name))
    kernel_name = 'python{}'.format(sys.version_info.major)

    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)
        ep.preprocess(nb, {'metadata':
                           {'path': os.path.abspath(ramp_kit_dir)}})


def convert_notebook(ramp_kit_dir='.'):
    problem_name = os.path.basename(os.path.abspath(ramp_kit_dir))
    print('Testing if the notebook can be converted to html')
    notebook_filename = os.path.join(
        os.path.abspath(ramp_kit_dir),
        '{}_starting_kit.ipynb'.format(problem_name))
    print(os.path.abspath(ramp_kit_dir))
    print('{}_starting_kit.ipynb'.format(problem_name))
    print(notebook_filename)
    notebook_html_filename = os.path.join(
        os.path.abspath(ramp_kit_dir),
        '{}_starting_kit.html'.format(problem_name))

    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
        nb_html, _ = nbconvert.export(HTMLExporter, nb)

    with open(os.path.join(os.path.abspath(ramp_kit_dir),
                           notebook_html_filename), 'wb') as f:
        f.write(nb_html.encode('utf-8'))

    delete_line_from_file(
        notebook_html_filename,
        '<link rel="stylesheet" href="custom.css">\n')
