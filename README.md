# The RAMP ecosystem

The [RAMP](http://www.ramp.studio) ecosystem contains two organizations and three libraries. The purpose of the bundle is to __define, build, manage, and optimize data analytics workflows__, typically on the top of open source machine learning libraries like [pandas](http://pandas.pydata.org), [scikit-learn](http://scikit-learn.org/), and [keras](https://github.com/fchollet/keras/tree/master/keras). The bundle consists of
1. [ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow) (this library) containing resuable tools and scripts to define
    1. [score types](rampwf/score_types) (metrics),
    2. [workflows and workflow elements](rampwf/workflows) (trainable data analytics modules like a classifier or a feature extractor),
    3. [cross-valudation schemes](rampwf/cv_schemes) (guiding the evaluation procedure of the workflow), and
    4. data connectors (to feed the workflows from various data sources).
2. [databoard](https://github.com/paris-saclay-cds/databoard), a library managing the frontend and the database of the [RAMP](http://www.ramp.studio) platform. (should may be renamed ramp-board)
3. [ramp-backend](https://github.com/paris-saclay-cds/ramp-backend), a library managing the RAMP backend (training and evaluating workflow instantiations aka submissions). (doesn't exist yet)
4. [ramp-data](https://github.com/ramp-data), an organization containing data sets on which workflows are trained and evaluated.
5. [ramp-kits](https://github.com/ramp-kits), an organization containing *starting kits*
    1. describing and implementing particular data analytics workflows, score types, cross validation schemes, and data connectors, using tools from [ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow), and
    2. implementing at least one workflow instantiation (submission) so the workflow can be unit tested.

## Why do I need want this bundle?

### I am a data science teacher

If you would like to **use one of the existing [ramp-kits](https://github.com/ramp-kits) and the corresponding data challenge in a classroom**, ask for a new event at the [RAMP site](http://www.ramp.studio/problems). You can browse the existing challenges by the [data science theme](http://www.ramp.studio/data_science_themes) you would like to focus on in your course, or by the [data domain](http://www.ramp.studio/data_domains) you would like to apply data science to.

If you have your own data set and would like to **build a new starting kit and challenge** for your course, go to ["Build your own workflow"](#build-your-own-workflow).

### I am a data science student / novice data scientits

You can **learn about data science** by signing up to ongoing and past data challenges at [ramp.studio](http://www.ramp.studio/problems). Sign up for the site then choose a [topic]((http://www.ramp.studio/data_science_themes) or a [data domain](http://www.ramp.studio/data_domains) and sign up to the corresponding event. Most events are in "open leaderboard" mode which means that you can **browse the code** of all the submissions, including the best ones submitted by 


## How to use this library?

### Build your own workflow

Chances are something similar already exists

### Launch your own RAMP


This library is part of the [RAMP](http://www.ramp.studio) ecosystem.

Toolkit for building analytics workflows on the top of pandas and scikit-learn. Primarily intended to feed RAMPs.

Workflow elements are file names. Most of them are python code files, they should have no extension. They will become editable on RAMP. Other files, e.g. external_data.csv or comments.txt whould have extensions. Editability fill be inferred from extension (e.g., txt is editable, csv is not, only uploadable). File names should contain no more than one '.'.

Tests suppose that ramp-kits and ramp-workflows are installed in the same directory.
