import os

from eralchemy import render_er

from ramp_database.utils import setup_db

from ramp_utils import read_config
from ramp_utils.testing import database_config_template


def main():
    database_config = read_config(
        database_config_template(), filter_section='sqlalchemy'
    )
    setup_db(database_config)

    render_er(
        "{}://{}:{}@{}:{}/{}"
        .format(
            database_config['drivername'], database_config['username'],
            database_config['password'], database_config['host'],
            database_config['port'], database_config['database']
        ),
        os.path.join('..', '_static', 'img', 'schema_db.png')
    )


if __name__ == '__main__':
    main()
