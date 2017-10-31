from __future__ import print_function

COLORS = {
    'black': '\x1b[30m',
    'red': '\x1b[31m',
    'green': '\x1b[32m',
    'yellow': '\x1b[33m',
    'blue': '\x1b[34m',
    'magenta': '\x1b[35m',
    'cyan': '\x1b[36m'}


def print_simple_color(text, color):
    """
    Pretty print with colors

    Parameters
    ----------
    text, color: str

    """
    try:
        color_key = color.lower()
        print("{}{}{}".format(COLORS[color_key], text, '\x1b[0m'))
    except KeyError:
        print(text)
