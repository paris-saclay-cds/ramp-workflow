.. _contributing:

Contributing
############

You are welcome to contribute to `ramp-workflow`_, particularly if there
are `Prediction types
<https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/prediction_types>`_,
`workflows
<https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/workflows>`_
or `score metrics
<https://github.com/paris-saclay-cds/ramp-workflow/tree/master/rampwf/score_types>`_
that you have written for your challenge which you think may be useful for
other challenges.

To contribute:

1. `Fork
   <https://help.github.com/en/github/getting-started-with-github/fork-a-repo>`_
   the `ramp-workflow`_ repository.
2. Clone the `ramp-workflow`_ repository then ``cd`` into it:

   .. code-block:: bash

    $ git clone https://github.com/paris-saclay-cds/ramp-workflow.git
    $ cd ramp-workflow

3. Install requirements:

   .. code-block:: bash

    $ pip install -r requirements.txt

Alternatively you may wish to install the required packages in a specific
environment for ramp-workflow.

4. Install ramp-worflow in editable mode:

   .. code-block:: bash

    pip install --editable .

5. Add your contributions and submit a `pull request
   <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests>`_
   to merge this into ramp-workflow.


.. _ramp-workflow: https://github.com/paris-saclay-cds/ramp-workflow

Release process
---------------

The following explain the main steps to release `ramp-board`:

1. Run `bumpversion release`. It will remove the `dev0` tag.
2. Commit the change `git commit -am "bumpversion 0.1.0"`.
3. Create a branch for this version `git checkout -b 0.1.X`.
4. Push the new branch into the upstream repository.
5. You can create a GitHub release.
6. Change the symlink in the `ramp-docs` repository such that stable point on
   0.1.X.
7. Push on PyPI by executing the following:
   `python setup.py sdist bdist_wheel && twine upload dist/*`.
8. In `master`, run `bumpversion minor`, commit and push on upstream.
