#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
back.py is a part of colored.

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


from .colors import names


class back(object):

    ESC = '\x1b[48;5;'
    END = 'm'
    num = 0
    for color in names:
        vars()[color] = '{}{}{}'.format(ESC, num, END)
        num += 1
