.. _using-kits:

Using RAMP starting-kits
########################

To get started working on an existing RAMP challenge:

1. Clone the `starting kit`_ for the challenge from GitHub:

   .. code-block:: bash
    
    $ git clone https://github.com/ramp-kits/<ramp_kit_name>.git
    $ cd <ramp_kit_name>

   If the starting kit does not include the data (i.e. a ``data/`` folder)
   you will need to download the data using the ``download_data.py`` file::

   .. code-block:: bash

    $ python download_data.py

2. Install dependencies:

   * with `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to create
     a virtual environment with all the required dependencies:

     .. code-block:: bash

        # to create the environment
        $ conda env create -f environment.yml
        # to activate the environment
        $ source activate <ramp_kit_name>

   * with `pip <https://pypi.org/project/pip/>`_:
     
     .. code-block:: bash

        $ pip install -r requirements.txt
        # install ramp-workflow
        $ pip install https://api.github.com/repos/paris-saclay-cds/ramp-workflow/zipball/master

3. Test that the starting kit works:

   .. code-block:: bash

    $ ramp_test_submission --quick-test

   Alternatively you can test the kit from a Python shell environment using::

    from rampwf.utils.testing import assert_submission
    assert_submission()    

   See the ``assert_submission()`` `source code
   <https://github.com/paris-saclay-cds/ramp-workflow/blob/master/rampwf/utils/testing.py#L63>`_
   for more details.

Now you are ready to write your own solution for the prediction problem.

Submiting to a RAMP event
=========================

To submit your solution to `RAMP studio`_:

1. Log in at `RAMP studio`_. If it is your first time, you will need to first
   register.
2. Find an open event for your RAMP challenge, e.g., the Titanic challenge.
   Sign up for the event. Note that registering for RAMP studio and signing
   up for events are controlled by RAMP administrators, so there can be a delay
   between asking to sign up and being able to submit.
3. Go to your sandbox and copy-paste/upload your solution script files, save,
   then submit. 
   
The submission is trained and tested on our backend in the same way as
``ramp_test_submission`` does it locally. When your submission is waiting in
the queue and being trained, you can find it in the
'New submissions (pending training)' table in 'my submissions'. Once it is
trained, you get will get an email, and your submission will show up on the
public leaderboard. If there is an error (note you should always test your
submission locally with ``ramp_test_submission``), it will show up in the
'Failed submissions' table in 'my submissions'. You can click on the error to
see part of the trace. The data set we use at the backend is usually different
from what you find in the starting kit, so the score may be different.

.. _starting kit: https://github.com/ramp-kits
.. _RAMP studio: http://www.ramp.studio
