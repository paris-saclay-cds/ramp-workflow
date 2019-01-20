import os
import ast

import click
import pandas as pd

from ...externals.tabulate import tabulate

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)


def _load_score_submission(submission_path, metric, step):
    """Load the score for a single submission."""
    training_output_path = os.path.join(submission_path, 'training_output')
    folds_path = [
        os.path.join(training_output_path, fold_name)
        for fold_name in os.listdir(training_output_path)
        if (os.path.isdir(os.path.join(training_output_path, fold_name)) and
            'fold_' in fold_name)
    ]
    data = {}
    for fold_id, path in enumerate(folds_path):
        score_path = os.path.join(path, 'scores.csv')
        if not os.path.exists(score_path):
            return
        scores = pd.read_csv(score_path, index_col=0)
        scores.columns.name = 'score'
        data[fold_id] = scores
    df = pd.concat(data, names=['fold'])
    metric = metric if metric else slice(None)
    step = step if step else slice(None)
    return df.loc[(slice(None), step), metric]


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """Command-line to show information about local submissions."""
    pass


@main.command()
@click.option("--ramp-kit-dir", default='.',
              help='Root directory of the ramp-kit to retrieved the train '
              'submission.')
@click.option("--metric", cls=PythonLiteralOption, default="[]",
              help='A list of the metric to report')
@click.option("--step", cls=PythonLiteralOption, default="[]",
              help='A list of the processing to report. Choices are '
              '{"train" , "valid", "test"}')
@click.option("--sort-by", cls=PythonLiteralOption, default="[]",
              help='Give the metric, step, and stat to use for sorting.')
@click.option("--ascending/--descending", default=True,
              help='Sort in ascending or descending order.')
@click.option("--precision", default=2,
              help='The precision for the different metrics reported.')
def leaderboard(ramp_kit_dir, metric, step, sort_by, ascending, precision):
    """Display the leaderboard for all the local submissions."""
    path_submissions = os.path.join(ramp_kit_dir, 'submissions')
    all_submissions = {
        sub: os.path.join(path_submissions, sub)
        for sub in os.listdir(path_submissions)
        if os.path.isdir(os.path.join(path_submissions, sub))
    }
    data = {}
    for sub_name, sub_path in all_submissions.items():
        scores = _load_score_submission(sub_path, metric, step)
        if scores is None:
            continue
        data[sub_name] = scores
    df = pd.concat(data, names=['submission'])
    df = df.unstack(level=['step'])
    df = pd.concat([df.groupby('submission').mean(),
                    df.groupby('submission').std()],
                   keys=['mean', 'std'], axis=1, names=['stat'])
    df = df.round(precision).reorder_levels([1, 2, 0], axis=1)
    step = ['train', 'valid', 'test'] if not step else step
    df = (df.sort_index(axis=1, level=0)
            .reindex(labels=step, level='step', axis=1))

    if sort_by:
        df = df.sort_values(tuple(sort_by), ascending=ascending, axis=0)

    headers = (["\n".join(df.columns.names)] +
               ["\n".join(col_names) for col_names in df.columns.get_values()])
    click.echo(tabulate(df, headers=headers, tablefmt='grid'))


def start():
    main()


if __name__ == '__main__':
    start()
