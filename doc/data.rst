.. _data:

Preparing your data
###################

The data for your RAMP challenge should consist of:

* **Private test data** - this data is stored on the RAMP server and is used to
  compute scores for each submission for the private leaderboard. It is
  essential that this data remains private from participants.
* **Private training data** - this data is also stored on the RAMP server and will
  be used by the server to train each submission. It is preferred that this
  is completely independent of the public data. However, if you have a small
  data size, it is also fine for this data to be the same as the public data.
* **Public data** - this data is made available to the participants. It needs to be
  split into 'public training data' and 'public testing' subsets. This is
  because the same script is used to test submissions when you run RAMP locally
  and on the RAMP server, when you make a submission to a RAMP event. Therefore,
  ``get_train_data()`` and ``get_test_data()`` from the ``problem.py`` script
  (see :ref:`Data I/O <in-out>`) needs to work
  both locally and on the RAMP server. This is generally achieved by having
  both a public test and train dataset and naming these the same as the private
  test and train dataset.
  See :ref:`prepare-ramp-data` for an example.

.. _prepare-ramp-data:

RAMP data
=========

A data directory is required to deploy your challenge on a RAMP sever.
Strictly speaking, it is not required if you simply wish to use RAMP workflow
locally. The data directory should be separate from the starting kit directory
(see :ref:`directory-structure`).

The data directory should consist of the following files, though technically,
only the data files are required for deployment.

* ``prepare_data.py`` - script to clean and split the raw data, ensuring that
  events can be deployed repeated with consistent data. See
  :ref:`prepare_data_script`.
* ``requirements.txt`` - file listing required packages used in
  ``prepare_data.py``.
* ``.travis.yml`` - Travis continuous integration configuration file. This
  should run the ``prepare_data.py`` and is set to run regularly so we are
  alerted to any problems. See the `configuration file
  <https://github.com/ramp-data/titanic/blob/master/.travis.yml>`_ of the
  Titanic challenge as an example.
* ``README.md`` - a quick summary of how to run ``prepare_data.py``, mostly
  serves as an introduction on GitHub.
* ``data/`` - directory that should ultimately contain the public and private
  datasets, after running ``prepare_data.py``. See :ref:`prepare_data_script`.

Generally, the data files for a RAMP challenge are kept in a repository
in the `ramp-data <https://github.com/ramp-data>`_ organisation on GitHub. This
is always a private repository and all data, public and private, can be kept
here, if size permits.

.. _prepare_data_script:

The prepare_data script
***********************

The ``prepare_data.py`` script should perform any data cleaning steps required
on the raw data and split the data into appropriate subsets as detailed above.
It is a good way to document all the data cleaning steps and enables you to
download (if required) and split the raw data easily on the RAMP server. It
also helps to ensure that same data challenge can be deployed again using
consistent training and test data subsets.

The raw data may be stored in the ``data/`` directory or can be downloaded from
elsewhere. Ultimately, there should be 4 data files in the ``data/``
directory, detailed below, after running ``prepare_data.py`` script.

As an example, the ``prepare_data.py`` for Titanic challenge, which stores the
raw data in ``data/``, is shown below::

    # 1 - read in or download the data.
    df_train = pd.read_csv(os.path.join('data', 'train.csv'))
    df_test = pd.read_csv(os.path.join('data', 'test.csv'))

    # 2 - Perform any data cleaning and split into private train/test subsets,
    # if required. Neither steps required in this case.

    # 3 - Split public train/test subsets. In this case the private training
    # data will be used as the public data
    df_public = df_train
    df_public_train, df_public_test = train_test_split(
        df_public, test_size=0.2, random_state=57)
        # specify the random_state to ensure reproducibility
    df_public_train.to_csv(os.path.join('data', 'public', 'train.csv'), index=False)
    df_public_test.to_csv(os.path.join('data', 'public', 'test.csv'), index=False)

Note that the private training data was also used as the public data, due to
the small size of this dataset. At this stage we have 4 files in the ``data/``
directory:

* ``train.csv`` - private training data.
* ``test.csv`` - private testing data. **This should never be made public.**
* ``public/train.csv`` - in this case, this was a subset of the private
  training data ``train.csv``.
* ``public/test.csv`` - in this case, this was a subset of the private training
  data ``train.csv``.

The public data files need to be copied over to the starting kit directory
(``ramp-kits/<starting_kit_name>/data/``, see :ref:`directory-structure`),
on a RAMP server, when deploying an event.

The data directory should look something like this::

    <starting_kit_name>/    # root data directory
    ├── README.md
    ├── requirements.txt
    ├── .travis.yml
    ├── prepare_data.py
    └── data/
        ├── train.csv     # any data file format acceptable
        ├── test.csv
        └── public/
            ├── train.csv
            └── test.csv

Strictly, only the ``data/`` directory is required to deploy an event on the
RAMP server, though it is good practice to include the other files.

See :ref:`directory-structure` for the structure of the data directory
relative to the starting kit directory.
