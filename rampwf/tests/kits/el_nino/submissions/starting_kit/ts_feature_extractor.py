import numpy as np

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120


def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    return tas.loc[:, lat_bottom:lat_top, lon_left:lon_right].mean(
        dim=('lat', 'lon'))


def get_enso_mean(tas):
    """The array of mean temperatures in the El Nino 3.4 region.

    At all time point.
    """
    return get_area_mean(
        tas, en_lat_bottom, en_lat_top, en_lon_left, en_lon_right)


class FeatureExtractor(object):

    def __init__(self):
        pass

    def transform(self, X_ds):
        """Compute the El Nino mean at time t - (12 - X_ds.n_lookahead).

        Corresponding the month to be predicted.
        """
        # This is the range for which features should be provided. Strip
        # the burn-in from the beginning.
        valid_range = np.arange(X_ds.n_burn_in, len(X_ds['time']))
        enso = get_enso_mean(X_ds['tas'])
        # Roll the input series back so it corresponds to the month to be
        # predicted
        enso_rolled = np.roll(enso, 12 - X_ds.n_lookahead)
        # Strip burn in.
        enso_valid = enso_rolled[valid_range]
        # Reshape into a matrix of one column
        X_array = enso_valid.reshape((-1, 1))
        return X_array
