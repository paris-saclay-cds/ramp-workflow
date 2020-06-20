import numpy as np

FEATURE_NAMES = [
    'theta_dot_2', 'theta_2', 'theta_dot_1', 'theta_1', 'torque',
    'cos_theta_1', 'sin_theta_1', 'cos_theta_2', 'sin_theta_2']


class FeatureExtractor:
    def __init__(self, restart_name, n_burn_in, n_lookahead):
        """
        Parameters
        ----------
        restart_name : str
            The name of the 0/1 column indicating restarts in the time series.
        """
        self.restart_name = restart_name
        self.n_burn_in = n_burn_in
        self.n_lookahead = n_lookahead

    def transform(self, X_df):
        """Transform time series into list of states.
        We use the observables at time t as the state

        Be careful not to use any information from the future (X_ds[t + 1:])
        when constructing X_df[t].
        Parameters
        ----------
        X_df : pandas DataFrame
            The raw time series.
        Return
        ------
        X_df : pandas Dataframe

        """
        # use copy to avoid SettingWithCopyWarning from pandas
        changed_df = add_sin_cos(X_df.copy())
        return changed_df[FEATURE_NAMES]


def add_sin_cos(df):
    """
    add cosine and sine of theta1 and theta2 to the dataframe
    :param df: input data
    :return: dataframe with cosine and sine features, those being cosine
    and sine of theta1 and theta2
    """
    thetas = ['theta_1', 'theta_2']
    for theta in thetas:
        df['cos_{}'.format(theta)] = np.cos(df[theta])
        df['sin_{}'.format(theta)] = np.sin(df[theta])
    return df
