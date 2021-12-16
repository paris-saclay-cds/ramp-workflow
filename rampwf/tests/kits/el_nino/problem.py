import os
import numpy as np
import pandas as pd
import xarray as xr
import rampwf as rw

problem_title = 'El Nino forecast'
Predictions = rw.prediction_types.make_regression()
# The time-series feature extractor step (the first step of the ElNino)
# workflow takes two additional parameters the parametrize the lookahead
# check. You may want to change these at the backend or add more check
# points to avoid that people game the setup.
workflow = rw.workflows.ElNino(check_sizes=[13], check_indexs=[13])
score_types = [
    rw.score_types.RMSE(name='rmse', precision=3),
]
# We do an 8-fold block cv. With the given parameters, we have
# length of common block: 300 months = 25 years
# length of validation block: 288 months = 24 years
# length of each cv block: 36 months = 3 years
cv = rw.cvs.TimeSeries(
    n_cv=3, cv_block_size=0.50, period=6, unit='month', unit_2='year')
get_cv = cv.get_cv


# Both train and test targets are stripped off the first
# n_burn_in entries
def _read_data(path, f_prefix):
    f_name = '{}.nc'.format(f_prefix)
    X_ds = xr.open_dataset(os.path.join(path, 'data', f_name))
    # making sure that time is not converted to object due to stupid
    # ns bug
    # note that this only works if time series has less then about 520 years
    X_ds['time'] = pd.date_range(
        '1/1/1700', periods=X_ds['time'].shape[0], freq='M')\
        - np.timedelta64(15, 'D')
    f_name = '{}.npy'.format(f_prefix)
    y_array = np.load(os.path.join(path, 'data', f_name))
    n_burn_in = X_ds.attrs['n_burn_in']
    return X_ds, y_array[n_burn_in:]


def get_train_data(path='.'):
    # return _read_data(path, 'train')
    return _read_data(path, 'train_mini')


def get_test_data(path='.'):
    # return _read_data(path, 'test')
    return _read_data(path, 'test_mini')
