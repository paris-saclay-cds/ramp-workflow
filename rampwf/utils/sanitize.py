"""
Utility for basic sanitation of user provided Python scripts.
False negatives are expected.
"""
import sys

BLACKLIST = [
    # subprocess module
    'subprocess', 'Popen',
    # shutil module
    'shutil', 'rmtree', 'copytree',
    # os module
    'fdopen', 'fchmod', 'fchown', 'ftruncate', 'open', 'listdir', 'scandir',
    'removedirs'
]


def _sanitize_input(code):
    """Sanitize the user provided Python code

    This is not intended to be failproof by any mean, but merely provide early
    warning / detections of users trying to tamper with the RAMP board system.
    """

    for key in BLACKLIST:
        if key in code:
            msg = "forbidden key word {} detected in submission.".format(key)
            if 'ramp_database' in sys.modules:
                msg += (
                    ' Tampering with the RAMP server is strictly forbidden! '
                    'Trying to cheat during a competition can cause '
                    'exclusion, legal consequences and/or a 0 grade in a '
                    'teaching course. The system administrator has been '
                    'informed. Please stop.'
                )
            raise RuntimeError(msg)
    return code
