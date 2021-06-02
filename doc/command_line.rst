.. _command-line:

RAMP-workflow commands
######################

The following commands are built using the
`click <https://click.palletsprojects.com/en/7.x/>`_ package which provides tab
completion for the command options. You however need to activate shell
completion by following the instructions given in the `click documentation <https://click.palletsprojects.com/en/7.x/bashcomplete/#activation>`_.
The `ramp-test` command also comes with tab completion for the submission name
if the submission you are looking for is located in the `./submissions/` folder.

.. click:: rampwf.utils.cli.testing:main
    :prog: ramp-test
    :show-nested:

.. click:: rampwf.utils.cli.show:main
    :prog: ramp-show
    :show-nested:
