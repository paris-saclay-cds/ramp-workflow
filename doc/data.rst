.. _data:

Preparing your data
###################

The data for your RAMP challenge should consist of:

* Private test data - this data is stored on the RAMP server and is used to
  compute scores for each submission for the private leaderboard. It is
  essential that this data remains private from participants.
* Private training data - this data is also stored on the RAMP server and will
  be used by the server to train each submission. It is good practice that this
  is completely independent of the public data. However, if you have a small
  data size, it is also fine for this data to be the same as the public data.
* Public data - this data is made available to the participants. It needs to be
  split into 'public training data' and 'public testing' subsets. This is
  because the same script is used to test submissions when you run RAMP locally
  and on the RAMP server, when you make a submission to a RAMP event. Therefore,
  ``get_train_data()`` and ``get_test_data()`` (see :ref:`in-out`) needs to work
  both locally and on the RAMP server. This is generally achieved by naming the
  public test and train dataset the same as the private test and train dataset.
  See :ref:`ramp-data` for an example.

.. _ramp-data:

RAMP-data
=========

By convention all the data files for a RAMP event are kept in a repository in
the `ramp-data <https://github.com/ramp-data>`_ organisation on GitHub. This
is always a private repository and all data, public and private, can be kept
here if size permits.

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
    df_public_train.to_csv(os.path.join('data', 'public_train.csv'), index=False)
    df_public_test.to_csv(os.path.join('data', 'public_test.csv'), index=False)

Note that the private training data was also used as the public data. At this
stage we have 4 files:

* ``train.csv`` - private training data.
* ``test.csv`` - private testing data. This should never be made public.
* ``public_train.csv`` - in this case, this was a subset of the private
  training data ``train.csv``.
* ``public_test.csv`` - in this case, this was a subset of the private training
  data ``train.csv``.

We now need to copy the public data files ``public_train.csv`` and
``public_test.csv`` into the 'ramp-kits' directory. Note that the script
assumes the file directory structure::

        <base_dir>
        ├── <ramp_kits_dir>/
        |   └── <ramp_name>/ # the starting kit would be in here
        |       └── data/
        └── <ramp_data_dir>/
            └── <ramp_name>/ # ramp data files and prepare_data.py file are here
                └── data/

The public files are being copied from the ``ramp_data_dir/data`` directory
into the ``ramp_kits_dir/data`` directory. They are also being renamed to
``train.csv`` and ``test.csv``, the same filenames as the private data in
``ramp_data_dir/data``. This is because, as mentioned above, the same script is
used to test submissions locally and on the RAMP server.

.. code-block:: python 

    # copy starting kit files to <ramp_kits_dir>/<ramp_name>/data
    copyfile(
        os.path.join('data', 'public_train.csv'),
        os.path.join(ramp_kits_dir, ramp_name, 'data', 'train.csv')
    )
    copyfile(
        os.path.join('data', 'public_test.csv'),
        os.path.join(ramp_kits_dir, ramp_name, 'data', 'test.csv')
    )

.. _download-data:

Downloading data
================

If your data is to be downloaded from elsewhere, you can simply download the
data in the ``prepare_data.py`` file and create the private and public datasets
on the server. You can also direct participants to download the public data
files from the RAMP server by providing a ``download_data.py`` in the starting
kit. This file should download the data when you open a terminal and run:

  .. code-block:: bash

    $ python download_data.py