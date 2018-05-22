#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
__init__.py is a part of colored.

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

from __future__ import print_function

from .colored import *
from .fore import *
from .back import *
from .style import *

__version_info__ = (1, 3, 5)
__version__ = '{0}.{1}.{2}'.format(*__version_info__)
