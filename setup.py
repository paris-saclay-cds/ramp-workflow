#! /usr/bin/env python

# Copyright (C) 2017 Balazs Kegl

import os
from codecs import open

from setuptools import setup, find_packages

import versioneer


# Get the long description from the README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()


descr = """Toolkit for building analytics workflows on the top of pandas and
scikit-learn. Primarily intended to feed RAMPs."""

DISTNAME = 'ramp-workflow'
DESCRIPTION = descr
MAINTAINER = 'Balazs Kegl'
MAINTAINER_EMAIL = 'balazs.kegl@gmail.com'
URL = 'https://github.com/paris-saclay-cds/ramp-workflow'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/paris-saclay-cds/ramp-workflow'


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(
        name=DISTNAME,
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        maintainer=MAINTAINER,
        include_package_data=True,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        url=URL,
        download_url=DOWNLOAD_URL,
        long_description=long_description,
        zip_safe=False,  # the package can run out of an .egg file
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS'],
        install_requires=[
            'numpy',
            'scipy',
            'pandas>=0.19.2',
            'scikit-learn>=0.18',
            'cloudpickle'],
        platforms='any',
        packages=find_packages(),
        entry_points={
            'console_scripts': [
                'ramp_test_submission='
                'rampwf.utils.command_line:ramp_test_submission',
                'ramp_test_notebook='
                'rampwf.utils.command_line:ramp_test_notebook',
                'ramp_convert_notebook='
                'rampwf.utils.command_line:ramp_convert_notebook',
            ]})
