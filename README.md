# The RAMP ecosystem

[![Build Status](https://travis-ci.org/paris-saclay-cds/ramp-workflow.svg?branch=master)](https://travis-ci.org/paris-saclay-cds/ramp-workflow)
[![Coverage](https://codecov.io/gh/paris-saclay-cds/ramp-workflow/branch/master/graphs/badge.svg?)](https://codecov.io/gh/paris-saclay-cds/ramp-workflow/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


The [RAMP][rstudio] ecosystem contains two organizations and two libraries. The purpose of the bundle is to __define, build, manage, and optimize data analytics workflows__, typically on the top of open source machine learning libraries like [pandas](http://pandas.pydata.org), [scikit-learn](http://scikit-learn.org/), and [keras](https://github.com/fchollet/keras). The bundle consists of

| Library/Organization | Purpose | Publicly available |
| :------ | :-----  | :------: |
| [ramp-workflow][rworkflow] | A set of reusable tools and scripts to define [score types](rampwf/score_types) (metrics), [workflow elements](rampwf/workflows), [prediction types](rampwf/prediction_types) and data connectors. | :white_check_mark: |
| [ramp-board][rboard] |  A library managing the frontend and the database of the [RAMP][rstudio] platform. | :white_check_mark: |
| [ramp-data][rdata] | An organization containing data sets on which workflows are trained and evaluated. | :no_entry_sign: |
| [ramp-kits][rkits] | An organization containing *starting kits* that use tools from [ramp-workflow][rworkflow] to implement a first valid (tested) workflow. | :white_check_mark: |

# Documentation

The RAMP-workflow documentation can be found 
[here](https://paris-saclay-cds.github.io/ramp-workflow/index.html). This will
detail who may be interested in the RAMP bundle and how to use RAMP workflow.

## Quick start

1. Install the latest `ramp-workflow` library 

```bash
$ pip install https://api.github.com/repos/paris-saclay-cds/ramp-workflow/zipball/master
```

This will set up some command line scripts like `ramp_test_submission`.
We suggest to use a dedicated virtual environment if you are familiar with it.

2. Pick a starting-kit on <https://github.com/ramp-kits>

Clone it locally and fire up the starting kit notebook.  
It will guide you through the problem, describe the data and the workflow, and let you run the pipeline.

For more details, visit the [documentation page](https://paris-saclay-cds.github.io/ramp-workflow/index.html).

## Contribute to [ramp-workflow][rworkflow]

`ramp-workflow` is meant to be a collaborative library. We value external contributions. 
Refer to our [contributing guide](https://paris-saclay-cds.github.io/ramp-workflow/contribute.html).

<!-- RAMP studio -->
[rstudio]: http://www.ramp.studio "RAMP main website"
[email]: mailto:admin@ramp.studio "Mailto: admin@ramp.studio"
[signup]: http://www.ramp.studio/sign-up "RAMP sign-up page"
[problems]: http://www.ramp.studio/problems "List of past RAMP challenges"
[themes]: http://www.ramp.studio/data_science_themes "Data science themes"
[domains]: http://www.ramp.studio/data_domains "Data domains"

<!-- git repos -->
[rworkflow]: https://github.com/paris-saclay-cds/ramp-workflow "Define RAMP score, workflow and CV scheme"
[rboard]: https://github.com/paris-saclay-cds/ramp-board "RAMP frontend library"
[rdata]: https://github.com/ramp-data "Organization for RAMP open data sets"
[rkits]: https://github.com/ramp-kits "Organization for RAMP starting kits"
