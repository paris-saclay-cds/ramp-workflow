RAMP workflow
=============

RAMP workflow allows to define and run machine learning pipeline, documentations available here_.

.. _here: https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/

Testing ramp-board
------------------

Since ramp-board_ depends on ramp-workflow, any time that there is a PR in ramp-workflow, ramp-board_ should also be tested using the following sequence:
 1. environment.yml_ and environment_iris_kit.yml_ on the rampwf_test_ branch should be modified to pull ramp-workflow from the PR'd branch.
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
