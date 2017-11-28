import imp
import os


def import_file(module_path, filename):
    """
    Import a submission file based on the filename and submission path.abs

    Parameters
    ----------
    module_path : str
        The path to the submission directory
    filename : str
        The filename of the submitted workflow element.

    Returns
    -------
    submitted_module : module object

    """
    submitted_path = ('.'
        .join(list(os.path.split(module_path)) + [filename])
        .replace('/', ''))
    submitted_file = '{}/{}.py'.format(module_path, filename)
    submitted_module = imp.load_source(submitted_path, submitted_file)
    return submitted_module
