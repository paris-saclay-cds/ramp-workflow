import click

from ..hyperopt import run_hyperopt

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--submission', default='starting_kit', show_default=True,
              help='The kit to hyperopt. It should be located in the '
              '"submissions" folder of the starting kit.')
@click.option('--ramp-kit-dir', default='.', show_default=True,
              help='Root directory of the ramp-kit to hyperopt.')
@click.option('--ramp-data-dir', default='.', show_default=True,
              help='Directory containing the data. This directory should '
              'contain a "data" folder.')
@click.option('--ramp-submission-dir', default='submissions',
              show_default=True,
              help='Directory where the submissions are stored. It is the '
              'directory (typically called "submissions" in the ramp-kit) '
              'that contains the individual submission subdirectories.')
@click.option('--engine', default='random', show_default=True,
              help='The name of the hyperopt engine, e.g., "random".')
@click.option('--n-iter', default=10, show_default=True,
              help='The number of hyperopt iterations, inputted to the '
              'engine. The granularity is per cv fold, so if you want to '
              'fully test 7 hyperparameter combinations for example with the '
              'random engine and you have 8 CV folds, you should enter '
              '--n-iter 56')
@click.option('--save-best', is_flag=True, default=True,
              show_default=True,
              help='Specify this flag to create a <submission>_hyperopt '
              'in the "submissions" dir with the best submission.')
def main(submission, ramp_kit_dir, ramp_data_dir, ramp_submission_dir,
         engine, n_iter, save_best):
    """Hyperopt a submission."""
    run_hyperopt(
        ramp_kit_dir=ramp_kit_dir, ramp_data_dir=ramp_data_dir,
        ramp_submission_dir=ramp_submission_dir, submission=submission,
        engine_name=engine, n_iter=n_iter, save_best=save_best)


def start():
    main()


if __name__ == '__main__':
    start()
