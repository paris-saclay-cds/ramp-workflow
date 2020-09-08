import os
import warnings

import click

from ..testing import assert_notebook
from ..testing import assert_submission

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--submission', default='starting_kit', show_default=True,
              help='The kit to test. It should be located in the '
              '"submissions" folder of the starting kit. If "ALL", all '
              'submissions in the directory will be tested.')
@click.option('--ramp-kit-dir', default='.', show_default=True,
              help='Root directory of the ramp-kit to test.')
@click.option('--ramp-data-dir', default='.', show_default=True,
              help='Directory containing the data. This directory should '
              'contain a "data" folder.')
@click.option('--data-label', default=None, show_default=True,
              help='A label specifying the data in case the same submissions '
              'are executed on multiple datasets. If specified, '
              'problem.get_train_data and problem.get_test_data should '
              'accept a data_label argument. Typically they can deal with '
              'multiple datasets containing the data within the directory '
              'specified by --ramp-data-dir (default: ./data), for example '
              'using subdirectories ./data/<data_label>/. It is also '
              'the subdirectory of submissions/<submission>/training_output '
              'where results are saved if --save-output is used.')
@click.option('--ramp-submission-dir', default='submissions',
              show_default=True,
              help='Directory where the submissions are stored. It is the '
              'directory (typically called "submissions" in the ramp-kit) '
              'that contains the individual submission subdirectories.')
@click.option('--notebook', is_flag=True, show_default=True,
              help='Whether or not to test the notebook.')
@click.option('--quick-test', is_flag=True,
              help='Specify this flag to test the submission on a small '
              'subset of the data.')
@click.option('--pickle', is_flag=True,
              help='Specify this flag to pickle the submission after '
              'training.')
@click.option('--save-output', is_flag=True,
              help='Specify this flag to save predictions, scores, eventual '
              'error trace, and state after training.')
@click.option('--retrain', is_flag=True,
              help='Specify this flag to retrain the submission on the full '
              'training set after the CV loop.')
@click.option('--ignore-warning', is_flag=True,
              help='Will filters all warning and avoid to print them.')
def main(submission, ramp_kit_dir, ramp_data_dir, data_label,
         ramp_submission_dir, notebook, quick_test, pickle, save_output,
         retrain, ignore_warning):
    """Test a submission and/or a notebook before to submit on RAMP studio."""
    if quick_test:
        os.environ['RAMP_TEST_MODE'] = '1'

    if ignore_warning:
        warnings.simplefilter("ignore")

    if submission == "ALL":
        ramp_submission_dir = os.path.join(ramp_kit_dir, 'submissions')
        submission = [
            directory
            for directory in os.listdir(ramp_submission_dir)
            if os.path.isdir(os.path.join(ramp_submission_dir, directory))
        ]
    else:
        submission = [submission]

    for sub in submission:
        assert_submission(ramp_kit_dir=ramp_kit_dir,
                          ramp_data_dir=ramp_data_dir,
                          data_label=data_label,
                          ramp_submission_dir=ramp_submission_dir,
                          submission=sub,
                          is_pickle=pickle,
                          save_output=save_output,
                          retrain=retrain)

    if notebook:
        assert_notebook(ramp_kit_dir=ramp_kit_dir)


def start():
    main()


if __name__ == '__main__':
    start()
