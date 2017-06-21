from os import environ, makedirs
from os.path import expanduser, join, exists


def get_data_home(ramp_kits_home=None):
    """Return the path of the ramp-kits data dir.

    This folder is used to fetch the up-to-date ramp-kits

    By default the data dir is set to a folder named 'ramp-kits'
    in the user home folder.

    Alternatively, it can be set by the 'RAMP-KITS' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    """
    if ramp_kits_home is None:
        ramp_kits_home = environ.get('RAMP-KITS',
                                join('~', 'ramp-kits'))
        ramp_kits_home = expanduser(ramp_kits_home)
    if not exists(ramp_kits_home):
        makedirs(ramp_kits_home)

    return ramp_kits_home
