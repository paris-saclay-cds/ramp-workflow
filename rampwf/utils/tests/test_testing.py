import os
import shutil
from tempfile import mkdtemp
from rampwf.kits import fetch_ramp_kit
from rampwf.utils import assert_submission, assert_notebook
from rampwf.kits.ramp_kit import RAMP_KITS_AVAILABLE

os.environ['RAMP_TEST_MODE'] = '1'


def test_submission_all_kits():
    tmp_dir = mkdtemp()
    try:
        for kit in RAMP_KITS_AVAILABLE:
            kit_dir = fetch_ramp_kit(kit, ramp_kits_home=tmp_dir)
            assert_submission(
                ramp_kit_dir=kit_dir, ramp_data_dir=kit_dir,
                submission='starting_kit', is_pickle=True, save_y_preds=True)
            # testing assert_notebook on titanic
            if kit == 'titanic':
                assert_notebook(ramp_kit_dir=kit_dir)

    finally:
        shutil.rmtree(tmp_dir)
