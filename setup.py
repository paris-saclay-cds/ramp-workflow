#! /usr/bin/env python

# Copyright (C) 2017 Balazs Kegl

import codecs
import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('rampwf', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'ramp-workflow'
DESCRIPTION = ("Toolkit for building analytics workflows on the top data "
               "science ecosystem")
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'A. Boucaud, B. Kegl, G. Lemaitre, J. Van den Bossche'
MAINTAINER_EMAIL = 'balazs.kegl@gmail.com, guillaume.lemaitre@inria.fr'
URL = 'https://github.com/paris-saclay-cds/ramp-workflow'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/paris-saclay-cds/ramp-workflow'
VERSION = __version__  # noqa
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
INSTALL_REQUIRES = ['numpy', 'scipy', 'pandas', 'scikit-learn>=0.22', 'joblib',
                    'cloudpickle', 'click']
EXTRAS_REQUIRE = {
    'tests': ['pytest', 'pytest-cov'],
    'docs': ['sphinx', 'sphinx_rtd_theme', 'numpydoc', 'sphinx-click']
}

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        'console_scripts': [
            'ramp-blend = rampwf.utils.cli.blend:start',
            'ramp-test = rampwf.utils.cli.testing:start',
            'ramp-show = rampwf.utils.cli.show:start',
            'ramp-hyperopt = rampwf.hyperopt.cli.hyperopt:start',
        ]})
