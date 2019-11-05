import numpy as np
from abc import ABC, abstractmethod
from scipy.special import gamma

# The maximum numbers of parameters a distribution would need
MAX_PARAMS = 2


class AbstractDists(ABC):

    @staticmethod
    @abstractmethod
    def pdf(x, params):
        pass

    @staticmethod
    @abstractmethod
    def assert_params(params):
        pass

    @staticmethod
    @abstractmethod
    def sample(params):
        pass


class Normal(AbstractDists):

    @staticmethod
    def pdf(x, params):
        mean = params[:, 0]
        sd = params[:, 1]
        Normal.assert_params(params)
        var = sd ** 2
        denom = (2 * np.pi * var) ** .5
        diff = x - mean
        num = np.exp(-diff ** 2 / (2 * var))
        probs = num / denom
        return probs

    @staticmethod
    def assert_params(params):
        sd = params[:, 1]
        assert np.all(sd > 0), "Make sure all sigmas are positive " \
                               "(second parameter)"

    @staticmethod
    def sample(params):
        return np.random.normal(params[0], params[1])


class Uniform(AbstractDists):

    @staticmethod
    def pdf(x, params):
        a = params[:, 0]
        b = params[:, 1]
        Uniform.assert_params(params)
        probs = np.zeros(x.shape)
        probs = np.where(np.logical_and(x >= a, x <= b), 1. / (b - a), probs)
        return probs

    @staticmethod
    def assert_params(params):
        a = params[:, 0]
        b = params[:, 1]
        assert np.all(a < b), "Make sure all \"a\" (first parameter)" \
                              " are bigger than \"b\" (second parameter)"
    @staticmethod
    def sample(params):
        return np.random.normal(params[0], params[1])


class Beta(AbstractDists):

    @staticmethod
    def pdf(x, params):
        a = params[:, 0]
        b = params[:, 1]
        loc = params[:, 2]
        scale = params[:, 3]
        y = (x - loc) / scale
        Beta.assert_params(x, params)
        probs = gamma(a+b)/(gamma(a)*gamma(b)) * y**(a-1) * (1-y)**(b-1)
        return probs

    @staticmethod
    def assert_params(x, params):
        a = params[:, 0]
        b = params[:, 1]
        loc = params[:, 2]
        scale = params[:, 3]
        assert np.all(a > 0), "Make sure all \"a\" > 0"
        assert np.all(b > 0), "Make sure all \"b\" > 0"
        assert np.all(x > loc), "Make sure all \"x\" > \"loc\""
        assert np.all(x < loc + scale), "Make sure all \"x\" < \"loc\"+\"scale\""

    @staticmethod
    def sample(params):
        y = np.random.beta(params[0], params[1])
        loc = params[:, 2]
        scale = params[:, 3]
        x = y * scale + loc
        return x


class EmptyDist(AbstractDists):

    @staticmethod
    def pdf(x, params):
        return np.zeros(params.shape[0])

    @staticmethod
    def assert_params(params):
        pass

    @staticmethod
    def sample(params):
        raise RuntimeError("You should not sample from an empty distribution")


def distributions_dispatcher(d_type):
    distributions_dict = {
        -1: EmptyDist,
        0: Normal,
        1: Uniform,
        2: Beta
    }
    dist = distributions_dict.get(d_type)
    if dist is None:
        raise KeyError("%s not a valid distribution type." % d_type)
    return dist


def get_pdf_from_dist(y, types, params):
    # All the distributions for a given dimension and a given y are the same.
    # The value need to be repeated to still be present when doing CV
    d_type = types[0]
    dist = distributions_dispatcher(d_type)
    return dist.pdf(y, params)


def sample_from_dist(d_type, params):
    dist = distributions_dispatcher(d_type)
    return dist.sample(params)
