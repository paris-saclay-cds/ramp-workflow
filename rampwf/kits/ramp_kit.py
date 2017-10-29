from os.path import join

import os
import pip
from subprocess import call

from .base import get_data_home

BASE_RAMP_KIT_URL = 'https://github.com/ramp-kits/'

RAMP_KITS_AVAILABLE = ('kaggle_seguro',
                       'mars_craters',
                       'MNIST_simplified',
                       'MNIST',
                       'california_rainfall_test',
                       'boston_housing',
                       'iris',
                       'titanic',
                       'epidemium2_cancer_mortality',
                       'drug_spectra',
                       'air_passengers',
                       'el_nino',
                       'HEP_tracking',
                       'mouse_cytometry',
                       )


def fetch_ramp_kit(name_kit, ramp_kits_home=None):
    """Fetcher of RAMP kit.

    Parameters
    ----------
    name_kit : str,
        The name of the RAMP kit to be fetched.

    Returns
    -------
    git_repo_dir : str,
        The path were the ramp-kit has been cloned.

    """
    import git

    if name_kit not in RAMP_KITS_AVAILABLE:
        raise ValueError("The ramp-kit '{}' requested is not available."
                         " The available kits are {}.".format(
                             name_kit, RAMP_KITS_AVAILABLE))
    ramp_kits_home = get_data_home(ramp_kits_home=ramp_kits_home)

    git_repo_dir = join(ramp_kits_home, name_kit)
    git.Repo.clone_from(join(BASE_RAMP_KIT_URL, name_kit),
                        git_repo_dir)

    print("The '{}' ramp-kit has been downloaded in the folder {}.".format(
        name_kit, git_repo_dir))

    os.chdir(git_repo_dir)
    if os.path.isfile('requirements.txt'):
        pip.main(["install", "-r", "requirements.txt"])

    if os.path.isfile('download_data.py'):
        call("python download_data.py", shell=True)

    return git_repo_dir
