.. _build-challenge:

Build your own RAMP challenge
#############################

If you have a new data set and a predictive problem, you may wish to use RAMP
to organise local experiments within your team, log model prototypes
and their scores or formalise your problem with resuable workflow building
blocks. If you are a :ref:`researcher <researcher-domain-science>` you may
be interested in setting up a RAMP challenge on `RAMP studio`_ to incite data
scientists and machine learning researchers to provide machine learning
solutions to your predictive problem. In this section we walk you through what
you need to do to either use RAMP workflow locally or to launch a RAMP
challenge on RAMP studio.

First, you will need to prepare your data by cleaning, if necessary, and
splitting it into the required public and private, test and training sets.
See :ref:`data` for more details.

Next, to setup your predictive problem to use RAMP workflow, the following
files/folders are required:

* ``problem.py`` - this parametrizes the setup and uses building blocks from
  RAMP workflow. More information about writing this script can be found at
  :ref:`problem`.
* ``submissions/`` - each solution to be tested should be stored in its own
  directory within ``submissions/``. The name of this new directory will serve
  as the ID for the submission. If you wish to launch a RAMP chellenge you
  will need to provide an example solution within ``submissions/starting_kit/``.
  Even if you are not launching a RAMP challenge on `RAMP Studio`_, it is
  useful to have an example submission as it shows which files are required,
  how they need to be named and how each file should be structured.
* data files - the data files of the challenge can be stored with your starting
  kit in a folder named ``data/``. Alternatively, your data may also be
  downloaded from elsewhere. If this is the case, you will need to provide a
  ``download_data.py`` file. This file should download the data when you open a
  terminal and run:

  .. code-block:: bash

    $ python download_data.py

Once you have the above files, it is quite easy to prepare the additional files
required for a full RAMP 'starting kit'. These files are not
required for RAMP workflow to function locally but are useful for participants
and are required to launch a RAMP challenge on `RAMP Studio`_.

* starting kit notebook - this is a jupyter notebook that introduces the
  predictive problem, provides some background information, exploratory
  data analysis and data visualisation, explains the workflow and provides a
  simple example solution. This example solution should generally be the same
  solution as within the ``submissions/starting_kit`` (see above).
* ``requirements.txt`` - lists the required packages, for participants that
  wish to use ``pip``.
* ``environment.yml`` - lists the required packages, for participants that wish
  to use ``conda``.
* ``README.md`` - this is the homepage when the challenge is on GitHub and
  should provide a quick start guide.
  
The base directory of a full ramp-kit should thus look like::

    <ramp_kit_name>/    # root ramp-kit directory
    ├── README.md
    ├── download_data.py (optional)
    ├── problem.py
    ├── requirements.txt
    ├── <ramp_kit_name>_starting_kit.ipynb
    ├── data
    └── submissions/
        └── <starting_kit>/    

If you wish to launch a RAMP challenge on `RAMP Studio`_ you will need to
upload the full ramp-kit to `ramp-kits <https://github.com/ramp-kits>`_.

.. _RAMP Studio: https://ramp.studio/
