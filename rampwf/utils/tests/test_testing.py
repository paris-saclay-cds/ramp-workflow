import shutil
from os import listdir
from os.path import join, isdir
from tempfile import mkdtemp
from rampwf.kits import fetch_ramp_kit
from rampwf.utils import assert_submission
from rampwf.kits.ramp_kit import RAMP_KITS_AVAILABLE


def test_submission_all_kits():
    tmp_dir = mkdtemp()
    try:
        for kit in RAMP_KITS_AVAILABLE:
            kit_dir = fetch_ramp_kit(kit, ramp_kits_home=tmp_dir)

            ramp_submission_dir = join(kit_dir, 'submissions')
            submissions = [directory
                           for directory in listdir(ramp_submission_dir)
                           if isdir(join(ramp_submission_dir, directory))]

            for sub in submissions:
                assert_submission(ramp_kit_dir=kit_dir,
                                  ramp_data_dir=kit_dir,
                                  submission=sub)
    finally:
        shutil.rmtree(tmp_dir)
