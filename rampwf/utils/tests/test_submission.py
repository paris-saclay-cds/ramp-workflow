import os
import pytest
import pickle
import tempfile

from rampwf.utils.submission import (
    pickle_trained_model, unpickle_trained_model)


def test_pickle_trained_model():
    # check that False is returned if trained model cannot be pickled

    # object raising PicklingError when dumped
    class Unpicklable(object):

        def __reduce__(self):
            raise pickle.PicklingError("not picklable")

    tmpdir = tempfile.mkdtemp()
    tmpfile = 'tmp.pkl'

    is_pickled = pickle_trained_model(
        tmpdir, Unpicklable(), trained_model_name=tmpfile, 
        is_silent=True, check_if_can_be_unpickled=False)
    assert not is_pickled
    with pytest.raises(pickle.PicklingError):
        pickle_trained_model(
            tmpdir, Unpicklable(), trained_model_name=tmpfile, 
            is_silent=False, check_if_can_be_unpickled=False)

    tmpdir = tempfile.mkdtemp()
    tmpfile = 'tmp.pkl'
    is_pickled = pickle_trained_model(
        tmpdir, 1, trained_model_name=tmpfile, 
        is_silent=True, check_if_can_be_unpickled=True)
    assert is_pickled
    is_pickled = pickle_trained_model(
        tmpdir, None, trained_model_name=tmpfile, 
        is_silent=True, check_if_can_be_unpickled=True)
    assert not is_pickled


def test_unpickle_trained_model():
    # check that None is returned if trained model cannot be unpickled

    tmpdir = tempfile.mkdtemp()
    tmpfile = 'tmp.pkl'

    trained_model = unpickle_trained_model(
        tmpdir, trained_model_name=tmpfile)
    assert trained_model is None
    
    with open(os.path.join(tmpdir, tmpfile), 'w') as file:
        file.write('dummy')

    trained_model = unpickle_trained_model(
        tmpdir, trained_model_name=tmpfile)
    assert trained_model is None
    with pytest.raises(pickle.UnpicklingError):
        trained_model = unpickle_trained_model(
            tmpdir, trained_model_name=tmpfile, is_silent=False)


