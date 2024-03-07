import pytest

from rampwf.utils.sanitize import _sanitize_input


def test_sanitize_input():
    _sanitize_input('Harmess code')

    msg = "forbidden key word open detected"
    with pytest.raises(RuntimeError, match=msg):
        _sanitize_input("with open('test.txt', 'wr') as fh")

    with pytest.raises(RuntimeError, match=msg):
        _sanitize_input("f = os.open('test.txt', 'wr')")

    # Make sure this does not catch open functions from other libraries:
    _sanitize_input("im = Image.open('im001.jpg')")

    msg = "forbidden key word scandir detected"
    with pytest.raises(RuntimeError, match=msg):
        _sanitize_input("for _ in os.scandir()")
