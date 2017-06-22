import shutil
from tempfile import mkdtemp
from rampwf.kits import fetch_ramp_kit
from rampwf.utils import assert_submission
from rampwf.kits.ramp_kit import RAMP_KITS_AVAILABLE


def test_submission_all_kits():
    tmp_dir = mkdtemp()
    try:
        for kit in RAMP_KITS_AVAILABLE:
            kit_dir = fetch_ramp_kit(kit, ramp_kits_home=tmp_dir)
            assert_submission(ramp_kit_dir=kit_dir,
                              ramp_data_dir=kit_dir,
                              submission_name='*')
    finally:
        shutil.rmtree(tmp_dir)


def test_submission_with_list():
    tmp_dir = mkdtemp()
    try:
        kit = 'titanic'
        kit_dir = fetch_ramp_kit(kit, ramp_kits_home=tmp_dir)
        submission_name = ['starting_kit', 'random_forest_20_5']
        assert_submission(ramp_kit_dir=kit_dir,
                          ramp_data_dir=kit_dir,
                          submission_name=submission_name)
    finally:
        shutil.rmtree(tmp_dir)


def test_submission_with_str():
    tmp_dir = mkdtemp()
    try:
        kit = 'titanic'
        kit_dir = fetch_ramp_kit(kit, ramp_kits_home=tmp_dir)
        submission_name = 'starting_kit'
        assert_submission(ramp_kit_dir=kit_dir,
                          ramp_data_dir=kit_dir,
                          submission_name=submission_name)
    finally:
        shutil.rmtree(tmp_dir)
