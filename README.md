# The RAMP ecosystem

The [RAMP](http://www.ramp.studio) ecosystem contains two organizations and three libraries. The purpose of the bundle is to __define, build, manage, and optimize data analytics workflows__, typically on the top of open source machine learning libraries like [pandas](http://pandas.pydata.org), [scikit-learn](http://scikit-learn.org/), and [keras](https://github.com/fchollet/keras/tree/master/keras). The bundle consists of
1. [ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow) (this library) containing resuable tools and scripts to define
    1. [score types](rampwf/score_types) (metrics),
    2. [workflows and workflow elements](rampwf/workflows) (trainable data analytics modules like a classifier or a feature extractor),
    3. [cross-valudation schemes](rampwf/cv_schemes) (guiding the evaluation procedure of the workflow), and
    4. data connectors (to feed the workflows from various data sources).
2. [ramp-board](https://github.com/paris-saclay-cds/ramp-board), a library managing the frontend and the database of the [RAMP](http://www.ramp.studio) platform. (should may be renamed ramp-board)
3. [ramp-backend](https://github.com/paris-saclay-cds/ramp-backend), a library managing the RAMP backend (training and evaluating workflow instantiations aka submissions). (doesn't exist yet)
4. [ramp-data](https://github.com/ramp-data), an organization containing data sets on which workflows are trained and evaluated.
5. [ramp-kits](https://github.com/ramp-kits), an organization containing *starting kits*
    1. describing and implementing particular data analytics workflows, score types, cross validation schemes, and data connectors, using tools from [ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow), and
    2. implementing at least one workflow instantiation (submission) so the workflow can be unit tested.

## Why do I want this bundle?

### I am a data science teacher

If you would like to **use one of the existing [ramp-kits](https://github.com/ramp-kits) and the corresponding data challenge in a classroom**, ask for a new event at the [RAMP site](http://www.ramp.studio/problems). You can browse the existing challenges by the [data science theme](http://www.ramp.studio/data_science_themes) you would like to focus on in your course, or by the [data domain](http://www.ramp.studio/data_domains) you would like to apply data science to.

If you have your own data set and would like to **build a new starting kit and challenge** for your course, go to ["Build your own workflow"](#build-your-own-workflow).

### I am a data science student or novice data scientist

You can **learn about data science** by signing up to ongoing and past data challenges at [ramp.studio](http://www.ramp.studio/problems). Sign up for the site then choose a [topic]((http://www.ramp.studio/data_science_themes) or a [data domain](http://www.ramp.studio/data_domains) and sign up to the corresponding event. Most events are in "open leaderboard" mode which means that you can **browse the code** of all the submissions, including the best ones submitted by top students or professional data scientists.

### I am a practicing data scientist

You can **[build your own workflow](#build-your-own-workflow)** using the [ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow) library, following examples from [ramp-kits](https://github.com/ramp-kits). You can then **train and test your models locally** and keep track of them in a simple file structure. If you want to **collaborate with your fellow team members**, you can simply commit your kit and use git.

You can also use [ramp.studio](http://www.ramp.studio) to expose your kit either privately to your internal team or by lunching a public data challenge. If you are interested in these options, [contact us](mailto:admin@ramp.studio).

### I am a researcher in machine learning

You can **benchmark your new algorithm** against all our data challenges on [ramp.studio](http://www.ramp.studio/problems). You can start by downloading the starting kits from the repo in [ramp-kits](https://github.com/ramp-kits) that you would like to use for benchmark, and test your algorithm locally. You can then sign up at the [RAMP site](http://www.ramp.studio) and sign up to one of the events corresponding to the kit you chose. You can submit your algorithm as many times as you want. You will have access to the public leaderboard score that uses cross validation described in the starting kit.

If you register with us for an official benchmarking, we will provide you a private test score for a small number of submissions of your choice, at a date of your choice (but only once).

### I am a researcher in a domain science

If you **have a predictive problem**, you can **submit it as a data challenge** to incite data scientists to solve your problem. First [build your own workflow](#build-your-own-workflow) using the [ramp-workflow](https://github.com/paris-saclay-cds/ramp-workflow) library, following examples from [ramp-kits](https://github.com/ramp-kits), then [contact us](mailto:admin@ramp.studio) so we upload it to the [RAMP site](http://www.ramp.studio). We can then organize hackatons and use the problem in a classroom setting. We may also automatically benchmark the thousands of models that are already in the platform.

## How to use this bundle?

Start by installing ramp-workflow (this library):
```
git clone https://github.com/paris-saclay-cds/ramp-workflow.git
cd ramp-workflow
python setup.py
```

### Get familiar with starting kits

Starting kits in [ramp-kits](https://github.com/ramp-kits) are working workflows and workflow instantiations. They work out of the box. You can run them using the `test_submission` script that simply executes [`test_submission.py`](rampwf/test_submission.py) in the starting kit. For example, clone the iris starting kit and test it by
```
mkdir ramp-kits
cd ramp-kits
git clone https://github.com/ramp-kits/iris.git
cd iris
test_submission
```
When `test_submission` is run without a parameter, it executes the workflow instantiation (submission) found in `submissions/starting_kit`. Iris uses a [`classifier`](rampwf/workflows/classifier) workflow which is instantiated by a single [`classifier.py`](https://github.com/ramp-kits/iris/blob/master/submissions/starting_kit/classifier.py) file in the submission directory (`submissions/starting_kit`). You can overwrite this file to test other classifiers, or keep it and make a new submission in the directory `submissions/<submission_name>`. You can then test this submission by executing `test_submission submission=<submission_name>`. For example,
```
test_submission submission=random_forest_10_10
```
will test [`classifier.py`](https://github.com/ramp-kits/iris/blob/master/submissions/random_forest_10_10/classifier.py) found in `submissions/random_forest_10_10`.

The starting kit also contains a Jupyter notebook named `<ramp_kit_name>_starting_kit.ipynb` (for example [`iris_starting_kit.ipynb`](https://github.com/ramp-kits/iris/blob/master/iris_starting_kit.ipynb)) that describes the predictive problem, the data set, and the workflow, and usually presents some exploratory analysis and data visualization.

### Submit at [ramp.studio](http://www.ramp.studio)

Once you found a good workflow instantiation (submission), you can submit it at [ramp.studio](http://www.ramp.studio). First, if it is your first time using RAMP, [sign up](http://www.ramp.studio/sign_up), otherwise [log in](http://www.ramp.studio/login). Then find an open event on the particular problem, for example, the event [iris_test](http://www.ramp.studio/events/iris_test) for iris. Sign up for the event. Both signups are controled by RAMP administrators, so there **can be a delay between asking for signup and being able to submit**.

Once your signup request is accepted, you can go to your [sandbox](http://www.ramp.studio/events/iris_test/sandbox) and copy-paste (or upload) `classifier.py` from `submissions/<submission_name>`. Save it, rename it, then submit it. The submission is trained and tested on our backend in the same way as `test_submission` does it locally. During your submission is waiting in the queue and being trained, you can find it in the "New submissions (pending training)" table in [my submissions](http://www.ramp.studio/events/iris_test/my_submissions). Once it is trained, you get a mail, and your submission shows up on the [public leaderboard](http://www.ramp.studio/events/iris_test/leaderboard). 
If there is an error (despite having tested your submission locally with `test_submission`), it will show up in the "Failed submissions" table in [my submissions](http://www.ramp.studio/events/iris_test/my_submissions). You can click on the error to see part of the trace.

After submission, do not forget to give credits to the previous submissions you reused or integrated into your submission.

The data set we use at the backend is usually different from what you find in the starting kit, so the score may be different.

The usual way to work with RAMP is to explore solutions, add feature transformations, select models, perhaps do some AutoML/hyperopt, etc., locally, and checking them with `test_submission`. The script prints mean cross-validation scores, for example, in the case of iris, 
```
----------------------------
train acc = 0.51 ± 0.043
train err = 0.49 ± 0.043
train nll = 1.21 ± 0.485
train f1_70 = 0.03 ± 0.1
valid acc = 0.47 ± 0.087
valid err = 0.53 ± 0.087
valid nll = 1.32 ± 0.686
valid f1_70 = 0.13 ± 0.221
test acc = 0.55 ± 0.131
test err = 0.45 ± 0.131
test nll = 0.87 ± 0.037
test f1_70 = 0.5 ± 0.167
```
The official score in iris (the first score column after "historical contributivity") is accuracy ("acc"), so the line that is relevant in the output of `test_submission` is `valid acc = 0.47 ± 0.087`. When the score is good enough, you can submit it at the RAMP.


### Build your own workflow

Chances are something similar already exists


Workflow elements are file names. Most of them are python code files, they should have no extension. They will become editable on RAMP. Other files, e.g. external_data.csv or comments.txt whould have extensions. Editability fill be inferred from extension (e.g., txt is editable, csv is not, only uploadable). File names should contain no more than one '.'.

Tests suppose that ramp-kits and ramp-workflows are installed in the same directory.
