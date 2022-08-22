#!/usr/bin/env bash
set -x
set -e

# Install dependencies with miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda

# create the environment
conda create --name testenv python=3.8
source activate testenv
pip install -r requirements.txt --progress-bar off
pip install "sphinx<5.0" sphinx_rtd_theme --progress-bar off
pip install sphinx-gallery sphinx-click
pip install numpydoc
pip install .

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make html | tee ~/log.txt

cd -
set +o pipefail
