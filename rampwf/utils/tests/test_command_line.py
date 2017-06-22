from rampwf.utils.command_line import create_parser


def test_cmd_parser():
    
    # defaults
    parser = create_parser()
    args = parser.parse_args([])
    assert args.ramp_kit_dir == '.'
    assert args.ramp_data_dir == '.'
    assert args.submission_name == 'starting_kit'

    # specifying keyword args
    parser = create_parser()
    args = parser.parse_args([
        '--ramp_kit_dir',  './titanic/', '--ramp_data_dir',  './titanic/',
        '--submission_name', 'other'])
    assert args.ramp_kit_dir == './titanic/'
    assert args.ramp_data_dir == './titanic/'
    assert args.submission_name == 'other'

