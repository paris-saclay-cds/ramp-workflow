"""
Utility to import local files from the filesystem as modules.

"""
import importlib


def import_module_from_source(source, name):
    """Load a module from a Python source file.

    Parameters
    ----------
    source : str
        Path to the Python source file which will be loaded as a module.
    name : str
        Name to give to the module once loaded.

    Returns
    -------
    module : Python module
        Return the Python module which has been loaded.
    """
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
