import numpy as np
from abc import ABC, abstractmethod
from scipy.special import gamma

# The maximum numbers of parameters a distribution would need
MAX_PARAMS = 5


class AbstractDists(ABC):
    nb_params = np.nan
    id = np.nan

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

    @staticmethod
    @abstractmethod
    def mu(params):
        pass

    @staticmethod
    @abstractmethod
    def var(params):
        pass


class Normal(AbstractDists):
    nb_params = 2
    id = 0

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

    @staticmethod
    def mu(params):
        return params[:, 0]

    @staticmethod
    def var(params):
        return params[:, 1] ** 2


class Uniform(AbstractDists):
    nb_params = 2
    id = 1

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

    @staticmethod
    def mu(params):
        a = params[:, 0]
        b = params[:, 1]
        return 0.5 * (a + b)

    @staticmethod
    def var(params):
        a = params[:, 0]
        b = params[:, 1]
        return (b - a) ** 2 / 12


class Beta(AbstractDists):
    nb_params = 4
    id = 2

    @staticmethod
    def pdf(x, params):
        a = params[:, 0]
        b = params[:, 1]
        loc = params[:, 2]
        scale = params[:, 3]
        y = (x - loc) / scale
        Beta.assert_params(x, params)
        probs = (gamma(a + b) * y ** (a - 1) * (1 - y) ** (b - 1)) / (gamma(a) * gamma(b))
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

    @staticmethod
    def mu(params):
        a = params[:, 0]
        b = params[:, 1]
        loc = params[:, 2]
        scale = params[:, 3]
        mu = a / (a + b)
        return mu * loc + scale

    @staticmethod
    def var(params):
        a = params[:, 0]
        b = params[:, 1]
        scale = params[:, 3]
        var = a * b / ((a + b) ** 2 * (a + b + 1))
        return var * scale ** 2


class EmptyDist(AbstractDists):
    nb_params = np.nan
    id = -1

    @staticmethod
    def pdf(x, params):
        return np.zeros(params.shape[0])

    @staticmethod
    def assert_params(params):
        pass

    @staticmethod
    def sample(params):
        raise RuntimeError("You should not sample from an empty distribution")

    @staticmethod
    def mu(params):
        raise RuntimeError("You should not get mu from an empty distribution")

    @staticmethod
    def var(params):
        raise RuntimeError("You should not get sigma from an empty distribution")


_distributions_dict = {
        cls.id: cls for cls in AbstractDists.__subclasses__()
    }


def distributions_dispatcher(d_type=-1):
    dist = _distributions_dict.get(d_type)
    if dist is None:
        raise KeyError("%s not a valid distribution type." % d_type)
    return dist


def sample_from_dist(d_type, params):
    dist = distributions_dispatcher(d_type)
    return dist.sample(params)
