from rampwf.utils.command_line import create_parser


def test_cmd_parser():
    
    # defaults
    parser = create_parser()
    args = parser.parse_args([])
    assert args.ramp_kit_dir == '.'
    assert args.ramp_data_dir == '.'
    assert args.submission == 'starting_kit'

    # specifying keyword args
    parser = create_parser()
    args = parser.parse_args([
        '--ramp_kit_dir',  './titanic/', '--ramp_data_dir',  './titanic/',
        '--submission', 'other'])
    assert args.ramp_kit_dir == './titanic/'
    assert args.ramp_data_dir == './titanic/'
    assert args.submission == 'other'

