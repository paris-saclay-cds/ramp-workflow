===============
Release history
===============
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`_
and this project adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

0.5.0 [Unreleased]
==================

0.4.0 - 2020-04-14
==================
Added
-----
- Autocompletion for submission names (`#261 <https://github.com/paris-saclay-cds/ramp-workflow/pull/261>`_)
- Add optional ``--data-label`` parameter to ``ramp_test```
  (`#245 <https://github.com/paris-saclay-cds/ramp-workflow/pull/245>`_)

Changed
-------
- Stop using colored prompt in Windows
  (`#224 <https://github.com/paris-saclay-cds/ramp-workflow/pull/224>`_)
- Use colored output only form terminals that know how to handle it
  (`#225 <https://github.com/paris-saclay-cds/ramp-workflow/pull/225>`_)
- Sanitize user input in workflows
  (`#226 <https://github.com/paris-saclay-cds/ramp-workflow/pull/226>`_)
- Set ``display.width`` to None in prompt
  (`#242 <https://github.com/paris-saclay-cds/ramp-workflow/pull/242>`_)
- Force utf8 encoding when reading notebook
  (`#252 <https://github.com/paris-saclay-cds/ramp-workflow/pull/252>`_)
- Handle the case of single hyperparameter in hyperopt
  (`#257 <https://github.com/paris-saclay-cds/ramp-workflow/pull/257>`_)

Fixed
-----

- Fix warnings in pickled models
  (`#233 <https://github.com/paris-saclay-cds/ramp-workflow/pull/233>`_)
- Create new Estimator classes to be compatible with ramp-board
  (`#222 <https://github.com/paris-saclay-cds/ramp-workflow/pull/222>`_)
- Pass all argument by position in `train_test_submission`
  (`add3fdc4c <https://github.com/paris-saclay-cds/ramp-workflow/commit/add3fdc4cd6afd1c42811616b1e10b7fed9be503>`_)

0.3.3 - 2020-03-13
==================

0.3.2 - 2020-02-28
==================

0.3.2 - 2020-02-28
==================

0.3.1 - 2020-02-01
==================

0.3.0 - 2020-01-30
==================

0.2.1 - 2020-01-21
==================

0.2.0 - 2020-01-21
==================

v0.2.0 (git tag only) - 2017-11-06
==================================
Added
-----
- refactored README and wiki
- ``ramp_test_submission`` script: beautified score printing, optional pickling of trained models, optional saving `y_pred`, cv bagging
- mars craters: detection workflow, detection scores,
- pollenating insects: a new simplified image classification workflow
- fake news: soft accuracy using a cost matrix
- kaggle seguro: normalized Gini score

v0.1.0 (git tag only) - 2017-10-10
==================================


[Unreleased]: https://github.com/paris-saclay-cds/ramp-workflow/compare/0.3.3...HEAD
