import os
import tempfile
import pickle

from rampwf.utils.submission import (
    pickle_trained_workflow, unpickle_trained_workflow)


def test_pickle_trained_workflow():
    # check that False is returned if trained workflow cannot be pickled

    # object raising PicklingError when dumped
    class Unpicklable(object):

        def __reduce__(self):
            raise pickle.PicklingError("not picklable")

    tmpdir = tempfile.mkdtemp()
    tmpfile = 'tmp.pkl'

    is_pickled = pickle_trained_workflow(
        tmpdir, Unpicklable(), trained_workflow_name=tmpfile)
    assert not is_pickled


def test_unpickle_trained_workflow():
    # check that None is returned if trained workflow cannot be unpickled

    tmpdir = tempfile.mkdtemp()
    tmpfile = 'tmp.pkl'

    trained_workflow = unpickle_trained_workflow(
        tmpdir, trained_workflow_name=tmpfile)
    assert trained_workflow is None
    
    with open(os.path.join(tmpdir, tmpfile), 'w') as file:
        file.write('dummy')

    trained_workflow = unpickle_trained_workflow(
        tmpdir, trained_workflow_name=tmpfile)
    assert trained_workflow is None


