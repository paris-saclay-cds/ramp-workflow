# coding: utf-8
"""
Utility methods to print the results in a terminal using term colors
"""
from __future__ import print_function

from pandas import option_context
from ..externals.colored import stylize, fg, attr

# Dictionary of term colors used for printing to terminal
fg_colors = {
    'official_train': 'light_green',
    'official_valid': 'light_blue',
    'official_test': 'red',
    'train': 'dark_sea_green_3b',
    'valid': 'light_slate_blue',
    'test': 'pink_1',
    'title': 'gold_3b',
    'warning': 'grey_46',
}


def print_title(str):
    print(stylize(str, fg(fg_colors['title']) + attr('bold')))


def print_warning(str):
    print(stylize(str, fg(fg_colors['warning'])))


def print_df_scores(df_scores, indent=''):
    """Pretty print the scores dataframe.

    Parameters
    ----------
    df_scores : pd.DataFrame
        the score dataframe
    indent : str, default=''
        indentation if needed
    """
    with option_context("display.width", 160):
        df_repr = repr(df_scores)
    df_repr_out = []
    for line, color_key in zip(df_repr.splitlines(),
                               [None, None] +
                               list(df_scores.index.values)):
        if line.strip() == 'step':
            continue
        if color_key is None:
            # table header
            line = stylize(line, fg(fg_colors['title']) + attr('bold'))
        if color_key is not None:
            tokens = line.split()
            tokens_bak = tokens[:]
            if 'official_' + color_key in fg_colors:
                # line label and official score bold & bright
                label_color = fg(fg_colors['official_' + color_key])
                tokens[0] = stylize(tokens[0], label_color + attr('bold'))
                tokens[1] = stylize(tokens[1], label_color + attr('bold'))
            if color_key in fg_colors:
                # other scores pale
                tokens[2:] = [stylize(token, fg(fg_colors[color_key]))
                              for token in tokens[2:]]
            for token_from, token_to in zip(tokens_bak, tokens):
                line = line.replace(token_from, token_to)
        line = indent + line
        df_repr_out.append(line)
    print('\n'.join(df_repr_out))
