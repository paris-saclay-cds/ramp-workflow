import os
import ast
import warnings

import click

from ..testing import blend_submissions

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:  # noqa: E722
            raise click.BadParameter(value)


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--ramp-kit-dir', default='.', show_default=True,
              help='Root directory of the ramp-kit to test.')
@click.option('--ramp-data-dir', default='.', show_default=True,
              help='Directory containing the data. This directory should '
              'contain a "data" folder.')
@click.option('--ramp-submission-dir', default='submissions',
              show_default=True,
              help='Directory where the submissions are stored. It is the '
              'directory (typically called "submissions" in the ramp-kit) '
              'that contains the individual submission subdirectories.')
@click.option('--data-label', default=None, show_default=True,
              help='A label specifying the data in case the same submissions '
              'are executed on multiple datasets. If specified, it is '
              'the subdirectory of submissions/<submission>/training_output '
              'where results are searched for blending.')
@click.option("--submissions", cls=PythonLiteralOption, default="'ALL'",
              show_default=True,
              help="""
              \bA list of the submissions to blend. Example:

              \b--submissions ['starting_kit','linear']
              """)
@click.option('--save-output', is_flag=True,
              help='Specify this flag to save predictions, scores, eventual '
              'error trace, and state after training.')
@click.option('--min-improvement', default=0.0, show_default=True,
              help='The minimum score improvement when adding'
              ' submissions to the ensemble.')
@click.option('--min-improvement', default=0.0, show_default=True,
              help='The minimum score improvement when adding'
              ' submissions to the ensemble.')
@click.option('--score-type-index', default=0, show_default=True,
              help='The index of the score type (from problem.py) on '
              'which the blending will be done.')
@click.option('--ignore-warning', is_flag=True,
              help='Will filters all warning and avoid to print them.')
def main(ramp_kit_dir, ramp_data_dir, ramp_submission_dir, data_label,
         submissions, save_output, min_improvement, score_type_index,
         ignore_warning):
    """Blend submissions."""

    if ignore_warning:
        warnings.simplefilter("ignore")

    if submissions == "ALL":
        submissions_dir = os.path.join(ramp_kit_dir, ramp_submission_dir)
        submissions = [
            directory
            for directory in os.listdir(submissions_dir)
            if data_label is None and os.path.isdir(os.path.join(
                submissions_dir, directory, 'training_output')) or
            data_label is not None and os.path.isdir(os.path.join(
                submissions_dir, directory, 'training_output', data_label))
        ]

    blend_submissions(ramp_kit_dir=ramp_kit_dir,
                      ramp_data_dir=ramp_data_dir,
                      ramp_submission_dir=ramp_submission_dir,
                      data_label=data_label,
                      submissions=submissions,
                      save_output=save_output,
                      min_improvement=float(min_improvement),
                      score_type_index=score_type_index)


def start():
    main()


if __name__ == '__main__':
    start()
