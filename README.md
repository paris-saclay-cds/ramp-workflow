# The RAMP ecosystem

The [RAMP][rstudio] ecosystem contains two organizations and three libraries. The purpose of the bundle is to __define, build, manage, and optimize data analytics workflows__, typically on the top of open source machine learning libraries like [pandas](http://pandas.pydata.org), [scikit-learn](http://scikit-learn.org/), and [keras](https://github.com/fchollet/keras). The bundle consists of

| Library/Organization | Purpose | Availability |
| :------ | :-----  | :------: |
| [ramp-workflow][rworkflow] | set of reusable tools and scripts to define [score types](rampwf/score_types) (metrics), [workflow elements](rampwf/workflows), [cross-validation schemes](rampwf/cv_schemes) and data connectors. | :white_check_mark: |
| [ramp-board][rboard] |  library managing the frontend and the database of the [RAMP][rstudio] platform. | :white_check_mark: |
| [ramp-backend][rbackend] | library managing the RAMP backend (training and evaluating workflow submissions). | :no_entry_sign: |
| [ramp-data][rdata] | organization containing data sets on which workflows are trained and evaluated. | :white_check_mark: |
| [ramp-kits][rkits] | organization containing *starting kits* that use tools from [ramp-workflow][rworkflow] to implement a first valid (tested) workflow. | :white_check_mark: |

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
[rbackend]: https://github.com/paris-saclay-cds/ramp-backend "RAMP backend library (not implemented)"
[rdata]: https://github.com/ramp-data "Organization for RAMP open data sets"
[rkits]: https://github.com/ramp-kits "Organization for RAMP starting kits"

## Why do I want this bundle?

### I am a data science teacher

If you would like to **use one of the existing [ramp-kits][rkits] and the corresponding data challenge in a classroom**, ask for a new event at the [RAMP site][problems]. You can browse the existing challenges by the [data science theme][themes] you would like to focus on in your course, or by the [data domain](http://www.ramp.studio/data_domains) you would like to apply data science to.

If you have your own data set and would like to **build a new starting kit and challenge** for your course, go to ["Build your own workflow"](#build-your-own-workflow).

### I am a data science student or novice data scientist

You can **learn about data science** by signing up to ongoing and past data challenges at [ramp.studio][problems]. Sign up for the site then choose a [topic][themes] or a [data domain][domains] and sign up to the corresponding event. Most events are in "open leaderboard" mode which means that you can **browse the code** of all the submissions, including the best ones submitted by top students or professional data scientists.

### I am a practicing data scientist

You can **[build your own workflow](#build-your-own-workflow)** using the [ramp-workflow][rworkflow] library, following examples from [ramp-kits][rkits]. You can then **train and test your models locally** and keep track of them in a simple file structure. If you want to **collaborate with your fellow team members**, you can simply commit your kit and use git.

You can also use [ramp.studio][rstudio] to expose your kit either privately to your internal team or by lunching a public data challenge. If you are interested in these options, [contact us][email].

### I am a researcher in machine learning

You can **benchmark your new algorithm** against all our data challenges on [ramp.studio][problems]. You can start by downloading the starting kits from the repo in [ramp-kits][rkits] that you would like to use for benchmark, and test your algorithm locally. You can then sign up at the [RAMP site][rstudio] and sign up to one of the events corresponding to the kit you chose. You can submit your algorithm as many times as you want. You will have access to the public leaderboard score that uses cross validation described in the starting kit.

If you register with us for an official benchmarking, we will provide you a private test score for a small number of submissions of your choice, at a date of your choice (but only once).

### I am a researcher in a domain science

If you **have a predictive problem**, you can **submit it as a data challenge** to incite data scientists to solve your problem. First [build your own workflow](#build-your-own-workflow) using the [ramp-workflow][rworkflow] library, following examples from [ramp-kits][rkits], then [contact us][email] so we upload it to the [RAMP site][rstudio]. We can then organize hackathons or longer data challenges, and use the problem in a classroom setting. We may also automatically benchmark the thousands of models that are already in the platform.

## How to use this bundle?

Start by installing ramp-workflow (this library):

```bash
git clone https://github.com/paris-saclay-cds/ramp-workflow.git
cd ramp-workflow
python setup.py
```

### Get familiar with starting kits

Starting kits in [ramp-kits][rkits] are working workflows and workflow instantiations. They work out of the box. You can run them using the [`test_submission`](bin/test_submission) script that simply executes [`test_submission.py`](rampwf/test_submission.py) in the starting kit. For example, clone the titanic starting kit and test it by

```bash
mkdir ramp-kits
cd ramp-kits
git clone https://github.com/ramp-kits/titanic.git
cd titanic
test_submission
```

When `test_submission` is run without a parameter, it executes the workflow instantiation (submission) found in `submissions/starting_kit`. Titanic uses a [`feature_extractor_classifier`](rampwf/workflows/feature_extractor_classifier.py) workflow which is instantiated by a [`feature_extractor.py`](https://github.com/ramp-kits/titanic/blob/master/submissions/starting_kit/feature_extractor.py) and a [`classifier.py`](https://github.com/ramp-kits/titanic/blob/master/submissions/starting_kit/classifier.py) file in the submission directory (`submissions/starting_kit`). You can overwrite these files to test other feature extractors and classifiers, or keep them and make a new submission in the directory `submissions/<submission_name>`. You can then test this submission by executing `test_submission submission=<submission_name>`. For example,

```shell
test_submission submission=random_forest_20_5
```

will test [`feature_extractor.py`](https://github.com/ramp-kits/titanic/blob/master/submissions/random_forest_20_5/feature_extractor.py) and [`classifier.py`](https://github.com/ramp-kits/titanic/blob/master/submissions/random_forest_20_5/classifier.py) found in `submissions/random_forest_20_5`.

The starting kit also contains a Jupyter notebook named `<ramp_kit_name>_starting_kit.ipynb` (for example [`titanic_starting_kit.ipynb`](https://github.com/ramp-kits/titanic/blob/master/titanic_starting_kit.ipynb)) that describes the predictive problem, the data set, and the workflow, and usually presents some exploratory analysis and data visualization.

### Submit to a data challenge at [ramp.studio][rstudio]

Once you found a good workflow instantiation (submission), you can submit it at [ramp.studio][rstudio]. First, if it is your first time using RAMP, [sign up][signup], otherwise [log in](http://www.ramp.studio/login). Then find an open event on the particular problem, for example, the event [titanic](http://www.ramp.studio/events/titanic) for this titanic. Sign up for the event. Both sign-ups are controlled by RAMP administrators, so there **can be a delay between asking to sign up and being able to submit**.

Once your signup request is accepted, you can go to your [sandbox](http://www.ramp.studio/events/titanic/sandbox) and copy-paste (or upload) [`feature_extractor.py`](https://github.com/ramp-kits/titanic/blob/master/submissions/starting_kit/feature_extractor.py) and [`classifier.py`](https://github.com/ramp-kits/titanic/blob/master/submissions/starting_kit/classifier.py) from `submissions/starting_kit`. Save it, rename it, then submit it. The submission is trained and tested on our backend in the same way as `test_submission` does it locally. During your submission is waiting in the queue and being trained, you can find it in the "New submissions (pending training)" table in [my submissions](http://www.ramp.studio/events/titanic/my_submissions). Once it is trained, you get a mail, and your submission shows up on the [public leaderboard](http://www.ramp.studio/events/titanic/leaderboard).
If there is an error (despite having tested your submission locally with `test_submission`), it will show up in the "Failed submissions" table in [my submissions](http://www.ramp.studio/events/titanic/my_submissions). You can click on the error to see part of the trace.

After submission, do not forget to give credits to the previous submissions you reused or integrated into your submission.

The data set we use at the backend is usually different from what you find in the starting kit, so the score may be different.

The usual way to work with RAMP is to explore solutions, add feature transformations, select models, perhaps do some AutoML/hyperopt, etc., _locally_, and checking them with `test_submission`. The script prints mean cross-validation scores
```
----------------------------
train auc = 0.85 ± 0.005
train acc = 0.81 ± 0.006
train nll = 0.45 ± 0.007
valid auc = 0.87 ± 0.023
valid acc = 0.81 ± 0.02
valid nll = 0.44 ± 0.024
test auc = 0.83 ± 0.006
test acc = 0.76 ± 0.003
test nll = 0.5 ± 0.005
```
The official score in titanic (the first score column after "historical contributivity" on the [leaderboard](http://www.ramp.studio/events/titanic/leaderboard)) is area under the ROC curve ("AUC"), so the line that is relevant in the output of `test_submission` is `valid auc = 0.87 ± 0.023`. When the score is good enough, you can submit it at the RAMP.

### Build your own workflow

If you are a [data science teacher](#i-am-a-data-science-teacher), a [data scientist](#i-am-a-practicing-data-scientist), or a [researcher](#i-am-a-researcher-in-a-domain-science) you may have a new data set and a predictive problem for which you want to build a starting kit. In this subsection we walk you through what you need to do.

Your goal is not necessary to launch an open RAMP, you may just want to organize your local experiments, make reusable building blocks, log your local submissions, etc. But once you have a working starting kit, it is also quite easy to launch a RAMP.

The basic gist is that each starting kit contains a python file `problem.py` that parametrizes the setup. It uses building blocks from this library ([ramp-workflow][rworkflow]), like choosing from a menu. As an example, we will walk you through the [`problem.py`](https://github.com/ramp-kits/titanic/blob/master/problem.py) of the titanic starting kit. Other problems may use more complex workflows or cross-validation schemes, but this complexity is usually hidden in the implementation of those elements in [ramp-workflow][rworkflow]. The goal was to keep the script `problem.py` as simple as possible.

#### 1. Choose a title.

```python
problem_title = 'Titanic survival classification'
```

#### 2. Choose a prediction type.

The prediction types are in [`rampwf/prediction_types`](rampwf/prediction_types)

```python
prediction_type = rw.prediction_types.multiclass
```

Typical prediction types are [`multiclass`](rampwf/prediction_types/multiclass.py) and [`regression`](rampwf/prediction_types/regression.py).

#### 3. Choose a workflow.

Available workflows are in [`rampwf/workflows`](rampwf/workflows).

```python
workflow = rw.workflows.FeatureExtractorClassifier()
```

Typical workflows are a single [`classifier`](rampwf/workflows/classifier.py) or a [feature extractor followed by a classifier](rampwf/workflows/feature_extractor_classifier.py) used here, but we have more complex workflows, named after the first problem that used them (e.g., [`drug_spectra`](rampwf/workflows/drug_spectra.py), two feature extractors, a classifier, and a regressor; or [`air_passengers`](rampwf/workflows/air_passengers.py), a feature extractor followed by a regressor, but also an `external_data.csv` that the feature extractor can merge with the training set). Each workflow implements a class which has `train_submission` and `test_submission` member functions that train and test submissions, and a `workflow_element_names` field containing the file names that `test_submission` expects in `submissions/starting_kit` or `submissions/<new-submission_name>`.

#### 4. Specify the prediction labels.

```python
prediction_labels = [0, 1]
```

If it is not a classification problem, set it to `None`.

#### 5. Choose score types.

Score types are metrics from [`rampwf/score_types`](rampwf/score_types)

```python
score_types = [
    rw.score_types.ROCAUC(name='auc', n_columns=len(prediction_labels)),
    rw.score_types.Accuracy(name='acc', n_columns=len(prediction_labels)),
    rw.score_types.NegativeLogLikelihood(name='nll',
                                         n_columns=len(prediction_labels)),
]
```

Typical score types are [`accuracy`](rampwf/score_types/accuracy.py) or [`RMSE`](rampwf/score_types/rmse.py). Each score type implements a class with a member function `score_function` and fields

  1. `name`, that `test_submission` uses in the logs; also the column name in the RAMP leaderboard,
  2. `precision`: the number of decimal digits,
  3. `n_columns`: the number of columns in the output `y_pred` of the last workflow element (typically a classifier or a regressor),
  4. `is_lower_the_better`: a boolean which is `True` if the score is the lower the better, `False` otherwise,
  5. `minimum`: the smallest possible score,
  6. `maximum`: the largest possible score.

#### 6. Write the cross-validation scheme.

Define a function `get_cv` returning a cross-validation object

```python
def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)
```

#### 7. Write the I/O methods.

The workflow needs two functions that read the training and test data sets.

```python
_target_column_name = 'Survived'
_ignore_column_names = ['PassengerId']


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
```

The convention is that these sets are found in `/data` and called `train.csv` and `test.csv`, but we kept this element flexible to accommodate a large number of possible input data connectors.

The script is used by [`test_submission.py`](rampwf/test_submission.py) which reads the files, implements the cross-validation split, instantiates the workflow with the submission, and trains and tests it. It is rather instructive to read this script to understand how we train the workflows. It is quite straightforward so we do not detail it here.

### Launching your own RAMP

In case you built your starting kit for launching a (public or private) data challenge, here are the additional steps to follow. In fact, these steps usually _precede_ the writing of the starting kit since we partition the data into public train and private test here.

#### 1. Create a repository.

The repository should be created typically in the [ramp-data](https://github.com/ramp-data) organization, which will hold your data set. It is important to keep the data private for allowing proper cross-validation and model testing. So either keep this repository private or make sure that the privacy of the data is assured using other techniques.

#### 2. Write your `prepare_data` script.

In the case of iris, [`prepare_data.py`](https://github.com/ramp-data/iris/blob/master/prepare_data.py) first reads the full data from `/data/iris.csv` and splits it into `/data/train.csv` and `/data/iris.csv`

```python
df = pd.read_csv(os.path.join('data', 'iris.csv'))
df_train, df_test = train_test_split(df, test_size=0.2, random_state=57)
df_train.to_csv(os.path.join('data', 'train.csv'), index=False)
df_test.to_csv(os.path.join('data', 'test.csv'), index=False)
```

`/data/test.csv` is the _private test_ data which is used to compute the scores on the private leaderboard, visible only to RAMP administrators. `/data/train.csv` is the _public train_ data on which we do cross validation to compute the scores on the public leaderboard. You do not need to follow this exact naming convention, what is important is that your convention matches what you do in the `problem.py` file of the corresponding starting kit, since, when we pull your data repository on the backend, we will test it with the same [`test_submission.py`](rampwf/test_submission.py) script as the script submitters use to test their submissions.

In the case of titanic, we already prepared train and test files so [`prepare_data.py`](https://github.com/ramp-data/titanic/blob/master/prepare_data.py) simply reads them here.

```python
df_train = pd.read_csv(os.path.join('data', 'train.csv'))
df_test = pd.read_csv(os.path.join('data', 'test.csv'))  # noqa
```

After preparing the backend data sets, we also usually prepare the public starting kit data sets that we will upload into the starting kit repo. It is a good practice to make the public data independent of both the training and test data on the backend, but it is also fine if the public data is the same as the backend training data (e.g., in case we don't have much data to spare), since "cheaters" can be caught by looking at their code and by them overfitting the public leaderboard. It is, on the other hand, crucial not to leak the private test data.

It is assumed that `ramp-kits` and `ramp-data` are installed in the same directory, but `prepare_data.py` also need to accept a `ramp_kits_dir` argument that specifies where to copy the public train and test files. In the case of iris, we do

```python
df_public = df_train
df_public_train, df_public_test = train_test_split(
    df_public, test_size=0.2, random_state=57)
df_public_train.to_csv(os.path.join('data', 'public_train.csv'), index=False)
df_public_test.to_csv(os.path.join('data', 'public_test.csv'), index=False)

# copy starting kit files to <ramp_kits_dir>/<ramp_name>/data
copyfile(
    os.path.join('data', 'public_train.csv'),
    os.path.join(ramp_kits_dir, ramp_name, 'data', 'train.csv')
)
copyfile(
    os.path.join('data', 'public_test.csv'),
    os.path.join(ramp_kits_dir, ramp_name, 'data', 'test.csv')
)
```

#### 3. Make sure the starting kit contains a Jupyter notebook

The notebook named `<ramp_kit_name>_starting_kit.ipynb`
(for example [`titanic_starting_kit.ipynb`](https://github.com/ramp-kits/titanic/blob/master/titanic_starting_kit.ipynb)) should describe the predictive problem, the data set, and the workflow, and usually presents some exploratory analysis and data visualization. This notebook will be rendered at the [RAMP site](http://www.ramp.studio/problems/titanic).

#### 4. [Send us a message][email].

In the backend, we will pull the data repo into `ramp-data` and the kit repo into `ramp-kits`, and test both with [`test_submission.py`](rampwf/test_submission.py). In the case of titanic,

```bash
mkdir ramp-data ramp-kits
git clone https://github.com/ramp-data/titanic.git ramp-data/titanic
git clone https://github.com/ramp-kits/titanic.git ramp-kits/titanic

python ramp-data/titanic/prepare_data.py

test_submission data=ramp-data/titanic path=ramp-kits/titanic
test_submission data=ramp-kits/titanic path=ramp-kits/titanic
```

### Contribute to [ramp-workflow][rworkflow]

It is possible that some of the elements (e.g., a score or a workflow) that you need for your starting kit is missing from `ramp-workflow`. First, look around, chances are something similar already exists. Second, you can implement it in your `problem.py` file, as we did with the cross validation object in [`titanic/problem.py`](https://github.com/ramp-kits/titanic/blob/master/problem.py). If you feel that the missing element can be useful in other problems, fork `ramp-workflow` and send us a pull request. Add a starting kit that uses the new element to the [`Makefile`](https://github.com/paris-saclay-cds/ramp-workflow/blob/readme/Makefile) as a unit test for the particular element.

<!---
# Draft
Most of the elements (submission files) are python code files, they should have no extension. They will become editable on RAMP. Other files, e.g. external_data.csv or comments.txt should have extensions. Editability fill be inferred from extension (e.g., txt is editable, csv is not, only uploadable). File names should contain no more than one '.'.



Tests suppose that ramp-kits and ramp-workflows are installed in the same directory.
-->
