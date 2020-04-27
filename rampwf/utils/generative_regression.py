import numpy as np
from abc import ABC, abstractmethod
from scipy.special import gamma, iv
from scipy.stats import truncnorm, norm, foldnorm, vonmises, beta

# The maximum numbers of parameters a distribution would need
# Only matters for bagging mixture models in
# prediction_types.generative_regression
MAX_MDN_PARAMS = 5


class AbstractDists(ABC):
    n_params = np.nan
    id = np.nan
    discrete = False
    params = None

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
    def mean(params):
        pass

    @staticmethod
    @abstractmethod
    def var(params):
        pass


class Normal(AbstractDists):
    n_params = 2
    id = 0
    params = ['mean', 'std']

    @staticmethod
    def pdf(x, params):
        mean = params[:, 0]
        std = params[:, 1]
        Normal.assert_params(params)
        var = std ** 2
        denom = (2 * np.pi * var) ** .5
        diff = x - mean
        num = np.exp(-diff ** 2 / (2 * var))
        probs = num / denom
        return probs

    @staticmethod
    def assert_params(params):
        std = params[:, 1]
        assert np.all(std > 0),\
            "Make sure all sigmas are positive (second parameter)"

    @staticmethod
    def sample(params):
        return np.random.normal(params[0], params[1])

    @staticmethod
    def mean(params):
        return params[:, 0]

    @staticmethod
    def var(params):
        return params[:, 1] ** 2


class Uniform(AbstractDists):
    n_params = 2
    id = 1
    params = ['a', 'b']

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
        return np.random.uniform(params[0], params[1])

    @staticmethod
    def mean(params):
        a = params[:, 0]
        b = params[:, 1]
        return 0.5 * (a + b)

    @staticmethod
    def var(params):
        a = params[:, 0]
        b = params[:, 1]
        return (b - a) ** 2 / 12


class Beta(AbstractDists):
    n_params = 4
    id = 2
    params = ['a', 'b', 'loc', 'scale']

    @staticmethod
    def pdf(x, params):
        a = params[:, 0]
        b = params[:, 1]
        loc = params[:, 2]
        scale = params[:, 3]
        y = (x - loc) / scale
        Beta.assert_params(params)
        probs = (gamma(a + b) * y ** (a - 1) * (1 - y) ** (b - 1)) / \
                (gamma(a) * gamma(b) * scale)

        probs = np.array(np.real(probs), dtype='float64')
        probs[x < loc] = 0
        probs[x > loc + scale] = 0
        return probs

    @staticmethod
    def assert_params(params):
        a = params[:, 0]
        b = params[:, 1]
        # loc = params[:, 2]
        scale = params[:, 3]
        assert np.all(a > 0), "Make sure all \"a\" > 0"
        assert np.all(b > 0), "Make sure all \"b\" > 0"
        assert np.all(scale > 0), "Make sure all \"scale\" > 0"
        # assert np.all(x >= loc), "Make sure all \"x\" > \"loc\""
        # assert np.all(x <= loc + scale), "Make sure all \"x\" < \"loc\"+\"scale\""

    @staticmethod
    def sample(params):
        y = np.random.beta(params[0], params[1])
        loc = params[:, 2]
        scale = params[:, 3]
        x = y * scale + loc
        return x

    @staticmethod
    def mean(params):
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


class NormalTruncated(AbstractDists):
    n_params = 4
    id = 3
    params = ['mean', 'std', 'a', 'b']

    @staticmethod
    def pdf(x, params):
        mean = params[:, 0]
        std = params[:, 1]
        a = params[:, 2]
        b = params[:, 3]
        NormalTruncated.assert_params(params)
        return truncnorm.pdf(x, a, b, mean, std)

    @staticmethod
    def assert_params(params):
        Normal.assert_params(params[:, :2])
        Uniform.assert_params(params[:, 2:])

    @staticmethod
    def sample(params):
        mean = params[:, 0]
        std = params[:, 1]
        a = params[:, 2]
        b = params[:, 3]
        return truncnorm.rvs(a, b, mean, std)

    @staticmethod
    def mean(params):
        mean = params[:, 0]
        std = params[:, 1]
        a = params[:, 2]
        b = params[:, 3]
        alpha = (a - mean) / sd
        beta = (b - mean) / sd
        return mean + std * ((norm.pdf(alpha) - norm.pdf(beta))
                             / (norm.cdf(beta) - norm.cdf(alpha)))

    @staticmethod
    def var(params):
        mean = params[:, 0]
        std = params[:, 1]
        a = params[:, 2]
        b = params[:, 3]
        alpha = (a - mean) / std
        beta = (b - mean) / std
        z = norm.cdf(beta) - norm.cdf(alpha)

        return (std ** 2) * (
            1 + ((alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / z) +
            ((norm.pdf(alpha) - norm.pdf(beta)) / z) ** 2
                            )


class NormalFolded(AbstractDists):
    n_params = 3
    id = 4
    params = ['c', 'loc', 'scale']

    @staticmethod
    def pdf(x, params):
        c = params[:, 0]
        loc = params[:, 1]
        scale = params[:, 2]
        NormalFolded.assert_params(params)
        return foldnorm.pdf(x, c, loc, scale)

    @staticmethod
    def assert_params(params):
        c = params[:, 0]
        assert np.all(c >= 0), "Make sure all \"c\" > 0"
        Normal.assert_params(params[:, 1:])

    @staticmethod
    def sample(params):
        c = params[:, 0]
        loc = params[:, 1]
        scale = params[:, 2]
        return foldnorm.rvs(c, loc, scale)

    @staticmethod
    def mean(params):
        raise NotImplementedError()

    @staticmethod
    def var(params):
        raise NotImplementedError()


class VonMises(AbstractDists):
    n_params = 3
    id = 5
    params = ['kappa', 'loc', 'scale']

    @staticmethod
    def pdf(x, params):
        kappa = params[:, 0]
        loc = params[:, 1]
        scale = params[:, 2]
        VonMises.assert_params(params)
        probs = vonmises.pdf(x, kappa, loc, scale)

        probs = np.array(probs)
        probs[x < loc - np.pi * scale] = 0
        probs[x > loc + np.pi * scale] = 0
        return probs

    @staticmethod
    def assert_params(params):
        kappa = params[:, 0]
        loc = params[:, 1]
        scale = params[:, 2]
        assert np.all(kappa > 0), "Make sure all \"kappa\" > 0"
        assert np.all(scale > 0), "Make sure all \"scale\" > 0"
        # assert np.all(x >= loc-np.pi*scale), "Make sure all \"x\" >= \"loc-np.pi*scale\""
        # assert np.all(x <= loc+np.pi*scale), "Make sure all \"x\" <= \"loc+np.pi*scale\""

    @staticmethod
    def sample(params):
        kappa = params[:, 0]
        loc = params[:, 1]
        scale = params[:, 2]
        return vonmises.rvs(kappa, loc, scale)

    @staticmethod
    def mean(params):
        return params[:, 1]

    @staticmethod
    def var(params):
        raise NotImplementedError()


class Pert(AbstractDists):
    n_params = 4
    id = 6
    params = ['a', 'b', 'c', 'lamb']

    @staticmethod
    def pdf(x, params):
        a = params[:, 0]
        b = params[:, 1]
        c = params[:, 2]
        lamb = params[:, 3]
        Pert.assert_params(params)

        size = c - a
        alphas = 1 + (lamb * ((b - a) / size))
        betas = 1 + (lamb * ((c - b) / size))
        x_trans = ((x - a) / size)

        probs = beta.pdf(x_trans, alphas, betas) / size
        probs = np.array(probs)
        probs[x < a] = 0
        probs[x > c] = 0
        return probs

    @staticmethod
    def assert_params(params):
        a = params[:, 0]
        b = params[:, 1]
        c = params[:, 2]
        lamb = params[:, 3]
        assert np.all(b > a), "Make sure all \"b\" > \"a\""
        assert np.all(c > b), "Make sure all \"c\" > \"b\""
        assert np.all(lamb > 0), "Make sure all \"lambda\" > 0"

    @staticmethod
    def sample(params):
        a = params[:, 0]
        b = params[:, 1]
        c = params[:, 2]
        lamb = params[:, 3]

        size = c - a
        alphas = 1 + lamb * (b - a) / size
        betas = 1 + lamb * (c - b) / size
        sampled = beta.rvs(alphas, betas) * size + a
        return sampled

    @staticmethod
    def mean(params):
        raise NotImplementedError()

    @staticmethod
    def var(params):
        raise NotImplementedError()


# https://en.wikipedia.org/wiki/Gaussian_function#Discrete_Gaussian
class NormalDiscrete(AbstractDists):
    n_params = 2
    id = 7
    discrete = True
    params = ['loc', 'scale']

    @staticmethod
    def pdf(x, params):
        loc = params[:, 0]
        scale = params[:, 1]
        NormalDiscrete.assert_params(params)
        probs = np.exp(-scale) * iv((x - loc), scale)
        probs[~np.equal(np.mod(x, 1), 0)] = 0
        return probs

    @staticmethod
    def assert_params(params):
        sd = params[:, 1]
        assert np.all(sd >= 0), "Make sure all sigmas are positive " \
                                "(second parameter)"


    @staticmethod
    def sample(params):
        # For sampling, how much do we look right and left of the mean
        tail = 5
        xmin = params[0] - (tail * params[1])
        xmax = params[0] - (tail * params[1])
        sampled = rejection_sampling(
            NormalDiscrete.pdf, params, xmin, xmax, discrete=True)
        return int(np.random.normal(params[0], params[1]))

    @staticmethod
    def mean(params):
        return params[:, 0]

    @staticmethod
    def var(params):
        return params[:, 1] ** 2


class EmptyDist(AbstractDists):
    n_params = np.nan
    id = -1
    params = None

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
    def mean(params):
        raise RuntimeError(
            "You should not get mean from an empty distribution")

    @staticmethod
    def var(params):
        raise RuntimeError(
            "You should not get sigma from an empty distribution")


distributions_dict = {
    cls.id: cls for cls in AbstractDists.__subclasses__()
}


# Bevington page 84.
# http://hosting.astro.cornell.edu/
# academics/courses/astro3310/Books/Bevington_opt.pdf
def rejection_sampling(
        pdf, params, xmin=0, xmax=1, discrete=False, nb_attemps=1000):
    if discrete:
        x = np.arange(xmin, xmax)
        x_sampler = np.random.randint
    else:
        x = np.linspace(xmin, xmax, nb_attemps)
        x_sampler = np.random.uniform
    y = pdf(x, params)
    pmin = 0.
    pmax = np.max(y)
    i = 0
    while i < nb_attemps:
        i += 1
        x = x_sampler(xmin, xmax)
        y = np.random.uniform(pmin, pmax)

        if y < pdf(x, params):
            return x
    return np.nan


def distributions_dispatcher(d_type=-1):
    dist = distributions_dict.get(d_type)
    if dist is None:
        raise KeyError("%s not a valid distribution type." % d_type)
    return dist


def sample_from_dist(d_type, params):
    dist = distributions_dispatcher(d_type)
    return dist.sample(params)
