[![Build Status](https://travis-ci.org/paris-saclay-cds/ramp-workflow.svg?branch=master)](https://travis-ci.org/paris-saclay-cds/ramp-workflow)
[![Coverage](https://codecov.io/gh/paris-saclay-cds/ramp-workflow/branch/master/graphs/badge.svg?)](https://codecov.io/gh/paris-saclay-cds/ramp-workflow/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Docs](https://img.shields.io/badge/docs-wiki-yellow.svg)](https://github.com/paris-saclay-cds/ramp-workflow/wiki)


# The RAMP ecosystem

The [RAMP][rstudio] ecosystem contains two organizations and two libraries. The purpose of the bundle is to __define, build, manage, and optimize data analytics workflows__, typically on the top of open source machine learning libraries like [pandas](http://pandas.pydata.org), [scikit-learn](http://scikit-learn.org/), and [keras](https://github.com/fchollet/keras). The bundle consists of

| Library/Organization | Purpose | Publicly available |
| :------ | :-----  | :------: |
| [ramp-workflow][rworkflow] | A set of reusable tools and scripts to define [score types](rampwf/score_types) (metrics), [workflow elements](rampwf/workflows), [prediction types](rampwf/prediction_types) and data connectors. | :white_check_mark: |
| [ramp-board][rboard] |  A library managing the frontend and the database of the [RAMP][rstudio] platform. | :no_entry_sign: |
| [ramp-data][rdata] | An organization containing data sets on which workflows are trained and evaluated. | :no_entry_sign: |
| [ramp-kits][rkits] | An organization containing *starting kits* that use tools from [ramp-workflow][rworkflow] to implement a first valid (tested) workflow. | :white_check_mark: |


## Why do I want this bundle ?

- [I am a data science teacher](https://github.com/paris-saclay-cds/ramp-workflow/wiki/I-am-a-data-science-teacher)
- [I am a data science student or novice data scientist](https://github.com/paris-saclay-cds/ramp-workflow/wiki/I-am-a-data-science-student)
- [I am a practicing data scientist](https://github.com/paris-saclay-cds/ramp-workflow/wiki/I-am-a-practicing-data-scientist)
- [I am a researcher in machine learning](https://github.com/paris-saclay-cds/ramp-workflow/wiki/I-am-a-machine-learning-researcher)
- [I am a researcher in a domain science or I have a predictive problem in my business](https://github.com/paris-saclay-cds/ramp-workflow/wiki/I-am-a-researcher-in-a-domain-science)

## Getting started

1. Install the latest `ramp-workflow` library 

```bash
$ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
```

This will set up some command line scripts like `ramp_test_submission`.
We suggest to use a dedicated virtual environment if you are familiar with it.

2. Pick up a starting-kit on <https://github.com/ramp-kits>

Clone it locally, and fire the starting kit notebook.  
It will guide you through the problem, describe the data and the workflow and let you run the pipeline.

Fore more details, visit the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki).

## Contribute to [ramp-workflow][rworkflow]

`ramp-workflow` is meant to be a collaborative library. We value external contributions. 
Refer to [this wiki page](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Contribute-to-ramp-workflow).

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
