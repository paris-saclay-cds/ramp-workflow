"""
Utility to import local files from the filesystem as modules.

"""
import importlib
from .sanitize import _sanitize_input


def import_module_from_source(source, name, sanitize=False):
    """Load a module from a Python source file.

    Parameters
    ----------
    source : str
        Path to the Python source file which will be loaded as a module.
    name : str
        Name to give to the module once loaded.
    sanitize: bool, default=False
        Check for blacklisted key words in code.

    Returns
    -------
    module : Python module
        Return the Python module which has been loaded.
    """
    if sanitize:
        with open(source, 'rt') as fh:
            _sanitize_input(fh.read())

    # import the module from the source file following the instructions given
    # in the importlib doc https://docs.python.org/3/library/importlib.html.
    # we do not add the module to sys.modules to make cloudpickle work.
    # see issue #232.
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
