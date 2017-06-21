import shutil
from tempfile import mkdtemp
from rampwf.kits import fetch_ramp_kit
from rampwf.utils import assert_submission

# later on we need to make a util to list kits to avoid repeating the same
# list everywhere
RAMP_KITS = ('boston_housing',
             'iris',
             'titanic',)


def test_submission_all_kits():
    tmp_dir = mkdtemp()
    try:
        for kit in RAMP_KITS:
            kit_dir = fetch_ramp_kit(kit, ramp_kits_home=tmp_dir)
            assert_submission(ramp_kit_dir=kit_dir,
                              ramp_data_dir=kit_dir,
                              submission_name='starting_kit')
    finally:
        shutil.rmtree(tmp_dir)
