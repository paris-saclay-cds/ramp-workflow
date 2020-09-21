# coding: utf-8
"""
Utility methods to print the results in a terminal using term colors
"""
import os
import platform

from pandas import option_context

from ..externals.colored import stylize, fg, attr

IS_WINDOWS = platform.system() == "Windows"
# known terminal types which can handle colors on any system
COLOR_TERMS = ['xterm-256color', 'cygwin', 'xterm-color']
# 'xterm' can handle color on macos but not on windows
IS_COLOR_TERM = 'TERM' in os.environ and (
    os.environ['TERM'] in COLOR_TERMS or (
        os.environ['TERM'] == 'xterm' and not IS_WINDOWS
    )
)

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


def print_title(title):
    if IS_COLOR_TERM:
        title = stylize(title, fg(fg_colors['title']) + attr('bold'))
    print(title)


def print_warning(warning):
    if IS_COLOR_TERM:
        warning = stylize(warning, fg(fg_colors['warning']))
    print(warning)


def print_df_scores(df_scores, indent=''):
    """Pretty print the scores dataframe.

    Parameters
    ----------
    df_scores : pd.DataFrame
        the score dataframe
    indent : str, default=''
        indentation if needed
    """
    with option_context("display.width", None):
        df_repr = repr(df_scores)
    df_repr_out = []
    for line, color_key in zip(df_repr.splitlines(),
                               [None, None] +
                               list(df_scores.index.values)):
        if line.strip() == 'step':
            continue
        if color_key is None:
            # table header
            if IS_COLOR_TERM:
                line = stylize(line, fg(fg_colors['title']) + attr('bold'))
        if color_key is not None:
            tokens = line.split()
            tokens_bak = tokens[:]
            if 'official_' + color_key in fg_colors:
                # line label and official score bold & bright
                if IS_COLOR_TERM:
                    label_color = fg(fg_colors['official_' + color_key])
                    tokens[0] = stylize(tokens[0], label_color + attr('bold'))
                    tokens[1] = stylize(tokens[1], label_color + attr('bold'))
            if IS_COLOR_TERM and (color_key in fg_colors):
                # other scores pale
                tokens[2:] = [stylize(token, fg(fg_colors[color_key]))
                              for token in tokens[2:]]
            for token_from, token_to in zip(tokens_bak, tokens):
                line = line.replace(token_from, token_to)
        line = indent + line
        df_repr_out.append(line)
    print('\n'.join(df_repr_out))
