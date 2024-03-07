"""
Utility for basic sanitation of user provided Python scripts.
False negatives are expected.
"""
import sys
import re

BLACKLIST = {
    # subprocess module
    'subprocess': 'subprocess',
    'Popen': 'Popen',
    # shutil module
    'shutil': 'shutil',
    'rmtree': 'rmtree',
    'copytree': 'copytree',
    # os module
    'fdopen': 'fdopen',
    'fchmod': 'fchmod',
    'fchown': 'fchown',
    'ftruncate': 'ftruncate',
    'listdir': 'listdir',
    'scandir': 'scandir',
    'removedirs': 'removedirs',
    # os.open or open keyword. Do not match all open as some challenge requires
    # opening Images for instance with PIL with Image.open.
    'open': '(?:os.| )open',
}


def _sanitize_input(code):
    """Sanitize the user provided Python code

    This is not intended to be failproof by any mean, but merely provide early
    warning / detections of users trying to tamper with the RAMP board system.
    """

    for kw, pattern in BLACKLIST.items():
        if len(re.findall(pattern, code)) > 0:
            msg = f"forbidden key word {kw} detected in submission."
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
