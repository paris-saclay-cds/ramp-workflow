from rampwf.utils.command_line import create_ramp_test_submission_parser
from rampwf.utils.command_line import create_ramp_test_notebook_parser


def test_cmd_ramp_test_submission_parser():

    # defaults
    parser = create_ramp_test_submission_parser()
    args = parser.parse_args([])
    assert args.ramp_kit_dir == '.'
    assert args.ramp_data_dir == '.'
    assert args.submission == 'starting_kit'

    # specifying keyword args
    parser = create_ramp_test_submission_parser()
    args = parser.parse_args([
        '--ramp_kit_dir', './titanic/', '--ramp_data_dir', './titanic/',
        '--submission', 'other'])
    assert args.ramp_kit_dir == './titanic/'
    assert args.ramp_data_dir == './titanic/'
    assert args.submission == 'other'


def test_cmd_ramp_test_notebook_parser():

    # defaults
    parser = create_ramp_test_notebook_parser()
    args = parser.parse_args([])
    assert args.ramp_kit_dir == '.'

    # specifying keyword args
    parser = create_ramp_test_notebook_parser()
    args = parser.parse_args(['--ramp_kit_dir', './titanic/'])
    assert args.ramp_kit_dir == './titanic/'
