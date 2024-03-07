from .base import BaseScoreType
import numpy as np
from sklearn.metrics import brier_score_loss


class BrierScore(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='brier_score', precision=3):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions):
        """A hybrid score.

        It tests the predicted _probability_ of the second class
        against the true _label index_ (which is 0 if the first label is the
        ground truth, and 1 if it is not, in other words, it is the
        true probability of the second class). Thus we have to override the
        `Base` function here
        """
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        return brier_score_loss(y_true_proba, y_proba)


class BrierSkillScore(BaseScoreType):
    is_lower_the_better = False
    minimum = -1.0
    maximum = 1.0

    def __init__(self, name='brier_score', precision=3):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions):
        """A hybrid score.

        It tests the predicted _probability_ of the second class
        against the true _label index_ (which is 0 if the first label is the
        ground truth, and 1 if it is not, in other words, it is the
        true probability of the second class). Thus we have to override the
        `Base` function here
        """
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        climo = np.ones(y_true_proba.size) * y_true_proba.mean()
        bs = brier_score_loss(y_true_proba, y_proba)
        bs_c = brier_score_loss(y_true_proba, climo)
        return 1 - bs / bs_c


class BrierScoreReliability(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='brier_score', precision=3,
                 bins=np.arange(0, 1.2, 0.1)):
        self.name = name
        self.precision = precision
        self.bins = bins
        self.bin_centers = (bins[1:] - bins[:-1]) * 0.05
        self.bin_centers[self.bin_centers > 1] = 1
        self.bin_centers[self.bin_centers < 0] = 0

    def score_function(self, ground_truths, predictions):
        """A hybrid score.

        It tests the predicted _probability_ of the second class
        against the true _label index_ (which is 0 if the first label is the
        ground truth, and 1 if it is not, in other words, it is the
        true probability of the second class). Thus we have to override the
        `Base` function here
        """
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        pos_obs_freq = np.histogram(
            y_proba[y_true_proba == 1], bins=self.bins)[0]
        fore_freq = np.histogram(y_proba, bins=self.bins)[0]
        pos_obs_rel_freq = np.zeros(pos_obs_freq.size)
        for p in range(pos_obs_rel_freq.size):
            if fore_freq[p] > 0:
                pos_obs_rel_freq[p] = pos_obs_freq[p] / fore_freq[p]
            else:
                pos_obs_rel_freq[p] = np.nan
        score = np.nansum(
            fore_freq * (self.bin_centers - pos_obs_rel_freq) ** 2)
        score /= float(y_proba.size)

        return score


class BrierScoreResolution(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='brier_score', precision=3,
                 bins=np.arange(0, 1.2, 0.1)):
        self.name = name
        self.precision = precision
        self.bins = bins
        self.bin_centers = (bins[1:] - bins[:-1]) * 0.05
        self.bin_centers[self.bin_centers > 1] = 1
        self.bin_centers[self.bin_centers < 0] = 0

    def score_function(self, ground_truths, predictions):
        """A hybrid score.

        It tests the predicted _probability_ of the second class
        against the true _label index_ (which is 0 if the first label is the
        ground truth, and 1 if it is not, in other words, it is the
        true probability of the second class). Thus we have to override the
        `Base` function here
        """
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        """
        See Murphy (1973) A vector partition of the probability score
        """
        np.seterr(divide="ignore")
        pos_obs_freq = np.histogram(
            y_proba[y_true_proba == 1], bins=self.bins)[0]
        fore_freq = np.histogram(y_proba, bins=self.bins)[0]
        climo = y_true_proba.mean()
        unc = climo * (1 - climo)
        pos_obs_rel_freq = np.zeros(pos_obs_freq.size)
        for p in range(pos_obs_rel_freq.size):
            if fore_freq[p] > 0:
                pos_obs_rel_freq[p] = pos_obs_freq[p] / fore_freq[p]
            else:
                pos_obs_rel_freq[p] = np.nan
        score = np.nansum(fore_freq * (pos_obs_rel_freq - climo) ** 2)
        score /= float(y_proba.size)
        return score / unc
