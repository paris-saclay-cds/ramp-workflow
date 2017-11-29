import os
import shutil
from tempfile import mkdtemp

import pytest

from rampwf.kits import fetch_ramp_kit
from rampwf.utils import assert_submission, assert_notebook
from rampwf.kits.ramp_kit import RAMP_KITS_AVAILABLE

os.environ['RAMP_TEST_MODE'] = '1'


@pytest.mark.parametrize('kit', RAMP_KITS_AVAILABLE)
def test_submission_all_kits(kit):
    tmp_dir = mkdtemp()
    try:
        kit_dir = fetch_ramp_kit(kit, ramp_kits_home=tmp_dir)
        # testing assert_notebook and optional switches on titanic
        if kit == 'titanic':
            assert_notebook(ramp_kit_dir=kit_dir)
            assert_submission(
                ramp_kit_dir=kit_dir, ramp_data_dir=kit_dir,
                submission='starting_kit', is_pickle=True,
                save_y_preds=True, retrain=True)
        else:
            assert_submission(
                ramp_kit_dir=kit_dir, ramp_data_dir=kit_dir,
                submission='starting_kit')

    finally:
        shutil.rmtree(tmp_dir)
