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
Strictly speaking, it is not required if you simply wish to use RAMP worflow
locally. The data directory should be separate from the starting kit directory
(see :ref:`directory-structure`).

The data directory should consist of the following files, though technically,
only the ``prepare_data.py`` file is required.

* ``prepare_data.py`` - script to clean and split the raw data, see
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

Generally, the data files for a RAMP challenge are kept in a repository
in the `ramp-data <https://github.com/ramp-data>`_ organisation on GitHub. This
is always a private repository and all data, public and private, can be kept
here, if size permits.

.. _prepare_data_script:

The prepare_data script
***********************

A ``prepare_data.py`` script should also be stored here. This script should
perform any data cleaning steps on the original data and split the data into
the appropriate subsets as detailed above. It is a good way to document all
the data cleaning steps. As an example, the Titanic challenge, which has
a very basic ``prepare_data.py`` file, is shown below::

    # In this case we have a predefined train/test cut so we are not splitting
    # the data here
    df_train = pd.read_csv(os.path.join('data', 'train.csv'))
    df_test = pd.read_csv(os.path.join('data', 'test.csv'))  # noqa

    # In this case the private training data was also used as the public data
    df_public = df_train
    df_public_train, df_public_test = train_test_split(
        df_public, test_size=0.2, random_state=57)
        # specify the random_state to ensure reproducibility
    df_public_train.to_csv(os.path.join('data', 'public', 'train.csv'), index=False)
    df_public_test.to_csv(os.path.join('data', 'public', 'test.csv'), index=False)

Note that the private training data was also used as the public data, due to
the small size of this dataset. At this stage we have 4 files:

* ``train.csv`` - private training data.
* ``test.csv`` - private testing data. **This should never be made public.**
* ``public/train.csv`` - in this case, this was a subset of the private
  training data ``train.csv``.
* ``public/test.csv`` - in this case, this was a subset of the private training
  data ``train.csv``.

The public data files should be copied over to the 'ramp kit' directory
when deploying an event on a RAMP server.

Data files
==========

In the Titanic example, the raw data was stored in the data directory. If your
data is to be downloaded from elsewhere, you can download the data in
the ``prepare_data.py`` file then clean and create the required private and
public datasets.
