.. _data:

Preparing your data
###################

The data for your RAMP challenge should consist of:

* Private test data - this data is stored on the RAMP server and is used to
  compute scores for each submission for the private leaderboard. It is
  essential that this data remains private from participants.
* Private training data - this data will be used by the server to train each
  submission. It is good practice that this is completely independent of the
  public data. However, if you have a small data size, it is also fine for this
  data to be the same as the public data.
* Public data - this data is made available to the participants. It needs to be
  split into 'public training data' and 'public testing' subsets. This is
  because the same script is used to calculate scores when you run RAMP locally
  and on the RAMP server, when you make a submission to a RAMP event. Therefore,
  ``get_train_data()`` and ``get_test_data()`` (see :ref:`in-out`) need to work
  locally and on the RAMP server. This is generally achieved by naming the
  public test and train dataset the same as the private test and train dataset.
  See :ref:`prepare-data` for an example.

.. _prepare-data:

The prepare_data.py file
========================



Downloading data
================

If this is the case provide a ``download_data.py``
  file that will download the data when you open a terminal and run:

  .. code-block:: bash

    $ python download_data.py