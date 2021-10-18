import tempfile
import pickle

from rampwf.utils.submission import pickle_model, unpickle_model


def test_pickle_model():
    # check that False is returned if trained workflow cannot be pickled

    # object raising PicklingError when dumped
    class Unpicklable(object):

        def __reduce__(self):
            raise pickle.PicklingError("not picklable")

    tmpdir = tempfile.mkdtemp()
    tmpfile = 'tmp.pkl'

    is_pickled = pickle_model(tmpdir, Unpicklable(), model_name=tmpfile)
    assert not is_pickled


def test_unpickle_model(capsys):
    # check that None is returned if trained workflow cannot be unpickled

    tmpdir = tempfile.mkdtemp()
    tmpfile = 'tmp.pkl'

    trained_workflow = unpickle_model(tmpdir, model_name=tmpfile)
    assert trained_workflow is None
