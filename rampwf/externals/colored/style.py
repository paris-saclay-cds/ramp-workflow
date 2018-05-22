#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
style.py is a part of colored.

Copyright 2014-2017 Dimitris Zlatanidis <d.zlatanidis@gmail.com>
All rights reserved.

Colored is very simple Python library for color and formatting in terminal.

https://github.com/dslackw/colored

colored is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""


class style(object):

    ESC = "\x1b["
    END = "m"

    names = {
        'BOLD': 1,
        'DIM': 2,
        'UNDERLINED': 4,
        'BLINK': 5,
        'REVERSE': 7,
        'HIDDEN': 8,
        'RESET': 0,
        'RES_BOLD': 21,
        'RES_DIM': 22,
        'RES_UNDERLINED': 24,
        'RES_BLINK': 25,
        'RES_REVERSE': 27,
        'RES_HIDDEN': 28
    }

    for color, num in names.items():
        vars()[color] = '{}{}{}'.format(ESC, num, END)
