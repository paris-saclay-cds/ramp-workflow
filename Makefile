PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-ctags:
	rm -f tags

clean: clean-ctags
	# $(PYTHON) setup.py clean
	rm -rf dist
	find . -name "*.pyc" | xargs rm -f

in: inplace # just a shortcut
inplace:
	# to avoid errors in 0.15 upgrade
	$(PYTHON) setup.py build_ext -i

test:
	nosetests --with-coverage rampwf/tests
	coverage run rampwf/test_submission.py path=$(RAMP_KITS_DIR)/boston_housing
	coverage run rampwf/test_submission.py path=$(RAMP_KITS_DIR)/boston_housing submission=random_forest_100
	coverage run rampwf/test_submission.py unknown_arg=try
	coverage run rampwf/test_submission.py path=$(RAMP_KITS_DIR)/boston_housing data=$(RAMP_DATA_DIR)/boston_housing
	coverage run rampwf/test_submission.py path=$(RAMP_KITS_DIR)/iris

test-all: test

trailing-spaces:
	find rampwf -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R rampwf

code-analysis:
	flake8 rampwf --ignore=E501,E211,E265 | grep -v __init__ | grep -v external
