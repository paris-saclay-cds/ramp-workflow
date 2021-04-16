PYTHON ?= python
PYTEST ?= pytest

all: clean inplace test

clean:
	 $(PYTHON) setup.py clean

install:
	pip install .

in: inplace # just a shortcut
inplace:
	pip install -e .

test-all:
	$(PYTEST) -vsl .

test: test-all

code-analysis:
	flake8 .

upload-pypi:
	python setup.py sdist bdist_wheel && twine upload dist/*

clean-dist:
	rm -r dist
