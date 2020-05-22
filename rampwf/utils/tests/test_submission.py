import tempfile
import pickle

from rampwf.utils.submission import pickle_model


def test_pickle_model(capsys):
    # check that warning is raised if trained workflow cannot be pickled

    # object raising PicklingError when dumped
    class Unpicklable(object):

        def __reduce__(self):
            raise pickle.PicklingError("not picklable")

    tmpdir = tempfile.mkdtemp()
    tmpfile = 'tmp.pkl'

    pickle_model(tmpdir, Unpicklable(), model_name=tmpfile)
    msg = "Warning: model can't be pickled."
    captured = capsys.readouterr()
    assert msg in captured.out
