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

Code style
----------

This repo uses `flake8` for code style. It can be run on commits automatically
by installing and activating `pre-commit <https://pre-commit.com/>`_:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Testing ramp-board
------------------

Since ramp-board_ depends on ramp-workflow, any time that there is a PR in ramp-workflow, ramp-board_ should also be tested using the following sequence:

 1. Under `ramp-board GitHub Actions <https://github.com/paris-saclay-cds/ramp-board/actions/workflows/main.yml>`_ in the `main` workflow select "Run Workflow" and in the dropdown menu paste the a pip installable URL to the version of ramp-workflow from your PR. For instance it would look as follows,::

      https://github.com/<your fork>/ramp-workflow/archive/refs/heads/<your branch>.zip


 2. If the tests are green, the PR'd branch of ramp-workflow can be accepted.
 3. If it is not, ramp-board_ should be fixed:

    1. Start a new branch in ramp-board_, preferably of the same name as the PR'd branch of ramp-workflow, on top of rampwf_test_.
    2. Fix the error on the new branch, make sure that the tests are green when pulling from ramp-workflow from the PR'd branch in environment.yml_ and environment_iris_kit.yml_.
    3. Accept the PR'd branch on ramp-workflow and update pypi.
    4. Modify environment.yml_ and environment_iris_kit.yml_ to pip install ramp-workflow from pypi and check again that the test are green.
    5. Accept the PR on ramp-board_.

.. _ramp-board: https://github.com/paris-saclay-cds/ramp-board
.. _environment.yml: https://github.com/paris-saclay-cds/ramp-board/blob/rampwf_test/environment.yml
.. _environment_iris_kit.yml: https://github.com/paris-saclay-cds/ramp-board/blob/rampwf_test/ci_tools/environment_iris_kit.yml
.. _rampwf_test: https://github.com/paris-saclay-cds/ramp-board/blob/rampwf_test

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
   `pip install build`
   `python -m build .`
   `twine upload dist/*`.
8. In `master`, run `bumpversion minor`, commit and push on upstream.
