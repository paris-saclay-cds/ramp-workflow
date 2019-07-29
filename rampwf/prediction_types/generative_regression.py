import numpy as np
from .base import BasePrediction
import itertools

def _regression_init(self, y_pred=None, y_true=None, n_samples=None):
    if y_pred is not None:
        self.y_pred = y_pred
    elif y_true is not None:
        self.y_pred = np.array(y_true)
    elif n_samples is not None:
        if self.n_columns == 0:
            shape = (n_samples, 1, 2 * self.n_bins + 1)
        else:
            shape = (n_samples, self.n_columns, 2 * self.n_bins + 1)
        self.y_pred = np.empty(shape, dtype=float)
        self.y_pred.fill(np.nan)
    else:
        raise ValueError(
            'Missing init argument: y_pred, y_true, or n_samples')


def make_generative_regression(n_bins, label_names=[]):
    """This additionally takes the number of bins as input"""

    Predictions = type(
        'GenerativeRegression',
        (BasePrediction,),
        {'label_names'   : label_names,
         'n_bins'        : n_bins,
         'n_columns'     : len(label_names),
         'n_columns_true': len(label_names),
         '__init__'      : _regression_init,
         # 'combine'       : combine, TODO: finish it and use it here to have a different ensembling method
         })
    return Predictions

@classmethod
def combine(cls, preds_list):


    label_names = preds_list[0].label_names

    if len(preds_list)==1:
        return cls(preds_list[0].y_pred)


    t_steps= len(preds_list[0].y_pred)
    reg_dims = len(preds_list[0].y_pred[0])

    new_preds=[]

    nbins = 0
    for reg in preds_list:
        nbins+=reg.n_bins + 1


    for i in range(t_steps):
        #for every timestep
        new_preds_t_step=[]
        for j in range(reg_dims):
            # We need to aggregate the prediction of time_dim to keep only one regressor per tick
            new_preds_dim_step = []

            new_bins= []
            selected = []
            for k, reg in enumerate(preds_list):
                bins_curr_reg = reg.y_pred[i,j,:reg.n_bins]
                new_bins.append(bins_curr_reg)
                selected.append([(k,l) for l in range(reg.n_bins)])
                probs_curr_reg = reg.y_pred[i, j, reg.n_bins:]

            new_bins = np.array(list(itertools.chain.from_iterable(new_bins)))
            selected = np.array(list(itertools.chain.from_iterable(selected)))
            idx_sorted= np.argsort(new_bins)
            new_bins = new_bins[idx_sorted]
            selected_sorted= selected[idx_sorted]

            new_preds_t_step.append(new_preds_dim_step)
        new_preds.append(new_preds_t_step)
