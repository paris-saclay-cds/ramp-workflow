.. _hyperopt:

The ramp hyperopt framework allows you to run
different hyperparameters optimization techniques, called here as **engines**.
The hyperparameters are defined in the submission kit.

To run a specific submission, you can use the `ramp-hyperopt` command line:

.. code-block:: bash

   $ ramp-hyperopt --engine random --submission rf --data-label cover_type_500 --n-trials 10 --save-best --label --test
