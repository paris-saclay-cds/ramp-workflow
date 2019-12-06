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
        except:  # noqa: E722
            raise click.BadParameter(value)


def _load_score_submission(submission_path, metric, step):
    """Load the score for a single submission."""
    training_output_path = os.path.join(submission_path, 'training_output')
    if not os.path.isdir(training_output_path):
        return None
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
@click.option("--ramp-kit-dir", default='.', show_default=True,
              help='Root directory of the ramp-kit to retrieved the train '
              'submission.')
@click.option("--metric", cls=PythonLiteralOption, default="[]",
              show_default=True,
              help='A list of the metric to report')
@click.option("--step", cls=PythonLiteralOption, default="[]",
              show_default=True,
              help='A list of the processing to report. Choices are '
              '{"train" , "valid", "test"}')
@click.option("--sort-by", cls=PythonLiteralOption, default="[]",
              show_default=True,
              help='Give the metric, step, and stat to use for sorting.')
@click.option("--ascending/--descending", default=True, show_default=True,
              help='Sort in ascending or descending order.')
@click.option("--precision", default=2, show_default=True,
              help='The precision for the different metrics reported.')
@click.option("--bagged/--mean", default=True, show_default=True,
              help='Bagged or mean scores.')
def leaderboard(ramp_kit_dir, metric, step, sort_by, ascending, precision,
                bagged):
    """Display the leaderboard for all the local submissions."""
    path_submissions = os.path.join(ramp_kit_dir, 'submissions')
    all_submissions = {
        sub: os.path.join(path_submissions, sub)
        for sub in os.listdir(path_submissions)
        if os.path.isdir(os.path.join(path_submissions, sub))
    }
    if bagged:  # bagged scores
        subs = []
        valid_scores = []
        test_scores = []
        for sub, path in all_submissions.items():
            bagged_scores_path = os.path.join(
                path, 'training_output', 'bagged_scores.csv')
            if not os.path.isfile(bagged_scores_path):
                continue
            bagged_scores_df = pd.read_csv(bagged_scores_path)
            n_folds = len(bagged_scores_df) // 2
            subs.append(sub)
            valid_scores.append(bagged_scores_df.iloc[n_folds - 1, 2])
            test_scores.append(bagged_scores_df.iloc[2 * n_folds - 1, 2])
            metric = bagged_scores_df.columns[2]
        df = pd.DataFrame()
        df['submission'] = subs
        df['valid {}'.format(metric)] = valid_scores
        df['test {}'.format(metric)] = test_scores
        headers = df.columns.get_values()
    else:  # mean scores with std
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
        df = df.reorder_levels([1, 2, 0], axis=1)
        step = ['train', 'valid', 'test'] if not step else step
        df = (df.sort_index(axis=1, level=0)
                .reindex(labels=step, level='step', axis=1))
        headers = (["\n".join(df.columns.names)] +
                   ["\n".join(col_names)
                    for col_names in df.columns.get_values()])

    df = df.round(precision)
    print(df.columns)
    if sort_by:
        df = df.sort_values(sort_by, ascending=ascending, axis=0)

    click.echo(tabulate(df, headers=headers, tablefmt='grid'))


def start():
    main()


if __name__ == '__main__':
    start()
