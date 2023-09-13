"""
Contains a number of test likelihoods and generates cobaya-models for them which can then
be called within test scripts etc.
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal, random_correlation
import scipy.stats
from numpy.random import default_rng
from cobaya.model import get_model

class Model_generator(metaclass=ABCMeta):
    """
    Base class for a model generator that all the other generators inherit from.
    """

    @abstractmethod
    def get_model(self):
        """
        returns a Cobaya model instance containing a prior and a likelihood (no sampler).
        """

    def auto_generate_parameter_names(self, input_params, dim):
        if input_params is None:
            self.input_params = [f"x_{d}" for d in range(dim)]
        else:
            assert len(input_params) == ndim, "Length of parameter name list doesn't match ndim"
            self.input_params = input_params

class Random_gaussian(Model_generator):
    """
    Randomly correlated gaussians with uniform prior.
    """
    def __init__(self, ndim=2, prior_size_in_std=5., random_mean_in_std=0.,
                 input_params=None):
        self.ndim = ndim
        self.prior_size_in_std = prior_size_in_std
        self.random_mean_in_std = random_mean_in_std
        self.mean = None
        self.std = None
        self.cov = None
        self.auto_generate_parameter_names(input_params, ndim)

    def redraw(self):
        rng = default_rng()
        self.std = rng.uniform(size=self.ndim)
        eigs = rng.uniform(size=self.ndim)
        eigs = eigs / np.sum(eigs) * self.ndim
        corr = random_correlation.rvs(eigs) if self.ndim > 1 else [[1]]
        self.cov = np.multiply(np.outer(self.std, self.std), corr)
        self.mean = rng.uniform(low=-1., size=self.ndim) * self.std * self.random_mean_in_std
        self.rv = multivariate_normal(self.mean, self.cov)

    def get_model(self):
        if self.mean is None:
            self.redraw()
        def gaussian(**kwargs):
            X = [kwargs[p] for p in self.input_params]
            return np.log(self.rv.pdf(X))
        info = {"likelihood": {"gaussian": {
            "external":gaussian, "input_params": self.input_params}}}
        info["params"] = {}
        for k, p in enumerate(self.input_params):
            p_max = (self.prior_size_in_std*self.std[k])
            p_min = (-self.prior_size_in_std*self.std[k])
            info["params"][p] = {"prior": {"min": p_min, "max": p_max}}
        model = get_model(info)
        return model

class Loggaussian(Random_gaussian):
    """
    Random gaussian with the first n directions in log-scale. The mean is set to 0 as
    standard. The method redraw will draw new random mean, stds and correlations.
    """
    def __init__(self, ndim=4, ndim_log=2, prior_size_in_std=5., random_mean_in_std=0.,
                 input_params=None):
        assert ndim >= ndim_log, "ndim needs to be larger than ndim_log"
        self.ndim_log = ndim_log
        super().__init__(ndim, prior_size_in_std, random_mean_in_std, input_params)

    def get_model(self):
        if self.mean is None:
            self.redraw()
        def loggaussian(**kwargs):
            X = [kwargs[p] for p in self.input_params]
            for j in range(self.ndim_log):
               X[j] = 10**X[j]
            return np.log(self.rv.pdf(X))
        info = {"likelihood": {"loggaussian": {
            "external":loggaussian, "input_params": self.input_params}}}
        info["params"] = {}
        for k, p in enumerate(self.input_params):
            p_max = (self.prior_size_in_std*self.std[k])
            p_min = (-self.prior_size_in_std*self.std[k])
            info["params"][p] = {"prior": {"min": p_min, "max": p_max}}
        model = get_model(info)
        return model

class Curved_degeneracy(Model_generator):
    """
    Curvy degeneracy with order 4 exponential family likelihood and uniform priors. Only
    available in 2d.
    """
    def __init__(self, a=10., b=0.45, c=4., d=20., bounds=None, input_params=None):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if bounds is None:
            self.bounds = np.array([[-0.5, 1.5], [-0.5, 2.]])
        else:
            self.bounds = bounds
        self.auto_generate_parameter_names(input_params, 2)

    def get_model(self):
        def curved_degeneracy(**kwargs):
            X = np.array([kwargs[p] for p in self.input_params])
            x_0 = X[..., 0]
            x_1 = X[..., 1]
            return  -(self.a*(self.b-x_0))**2./self.c - (self.d*(x_1/self.c-x_0**4.))**2.
        info = {"likelihood": {"curved_degeneracy": {
            "external":curved_degeneracy, "input_params": self.input_params}}}
        info["params"] = {}
        for k, p in enumerate(self.input_params):
            info["params"][p] = {"prior": {"min": self.bounds[k, 0], "max": self.bounds[k, 1]}}
        model = get_model(info)
        return model

class Ring(Model_generator):
    """
    A ring shaped likelihood. Only available in 2d
    """
    def __init__(self, mean_radius=1., std=0.05, offset=0., prior_size_in_std=5.,
                 input_params=None):
        self.mean_radius = mean_radius
        self.std = std
        self.offset = offset
        self.prior_size_in_std = prior_size_in_std
        self.auto_generate_parameter_names(input_params, 2)

    def get_model(self):
        def ring(**kwargs):
            X = np.array([kwargs[p] for p in self.input_params])
            x_0 = X[..., 0]
            x_1 = X[..., 1]
            return scipy.stats.norm.logpdf(np.sqrt(x_0**2 + x_1**2)+ \
                self.offset*x_0, loc=self.mean_radius, scale=self.std)
        info = {"likelihood": {"ring": {
            "external":ring, "input_params": self.input_params}}}
        lower_x_0 = self.offset - self.mean_radius - self.prior_size_in_std*self.std
        upper_x_0 = self.offset + self.mean_radius + self.prior_size_in_std*self.std
        lower_x_1 = -1*self.mean_radius - self.prior_size_in_std*self.std
        upper_x_1 = self.mean_radius + self.prior_size_in_std*self.std
        bounds = np.array([[lower_x_0, upper_x_0], [lower_x_1, upper_x_1]])
        info["params"] = {}
        for k, p in enumerate(self.input_params):
            info["params"][p] = {"prior": {"min": bounds[k, 0], "max": bounds[k, 1]}}
        model = get_model(info)
        return model

class Himmelblau(Model_generator):
    """
    The Himmelblau function. Usually used to test optimization algorithms. Although
    it is generalised to n dimensions be aware that it is only really used in 2 usually.
    If no bounds are provided they are set to (-4,4) in every dimension as standard.
    """

    def __init__(self, ndim=2, a=11., b=7., bounds=None, input_params=None):
        self.a = a
        self.b = b
        if bounds is None:
            self.bounds = np.array([[-4., 4.]]*ndim)
            print(self.bounds)
        else:
            self.bounds = bounds
        self.auto_generate_parameter_names(input_params, ndim)

    def get_model(self):
        def himmelblau(**kwargs):
            def himmel(x,y):
                return ((x*x+y-self.a)**2 + (x+y*y - self.b)**2)
            X = [kwargs[p] for p in self.input_params]
            if len(X)%2 ==0:
                chi2 = 0
                for i in range(0,len(X)//2,2):
                    chi2+=himmel(X[i],X[i+1])
                return -.5 * chi2
            else:
                chi2 = 0
                for i in range(0,len(X)//2,2):
                    chi2+=himmel(X[i],X[i+1])
                chi2 += X[-1]**2
                return -.5 * chi2
        info = {"likelihood": {"himmelblau":
            {"external": himmelblau, "input_params": self.input_params}}}
        info["params"] = {}
        for d, p_d in enumerate(self.input_params):
            info["params"][p_d] = {"prior":
                {"min": self.bounds[d, 0], "max": self.bounds[d, 1]}}
        model = get_model(info)
        return model

class Rosenbrock(Model_generator):
    """
    The Rosenbrock function. Usually used to test optimization algorithms. Although
    it is generalised to n dimensions be aware that it is only really used in 2 usually.
    If no bounds are provided they are set to (-4,4) in every dimension as standard.
    """

    def __init__(self, ndim=2, a=1., b=100., bounds=None, input_params=None):
        self.a = a
        self.b = b
        if bounds is None:
            self.bounds = np.array([[-4., 4.]]*ndim)
        else:
            self.bounds = bounds
        self.auto_generate_parameter_names(input_params, ndim)

    def get_model(self):
        def rosen(x,y):
            return (self.a-x)**2+self.b*(y-x*x)**2
        def rosenbrock(**kwargs):
            X = [kwargs[p] for p in self.input_params]
            if len(X)%2 ==0:
                chi2 = 0
                for i in range(0,len(X),2):
                    chi2+=rosen(X[i],X[i+1])
                return -0.5*chi2
            else:
                chi2 = 0
                for i in range(0,len(X)//2,2):
                    chi2+=rosen(X[i],X[i+1])
                chi2 += X[-1]**2
                return -.5 * chi2
        info = {"likelihood": {"rosenbrock":
            {"external": rosenbrock, "input_params": self.input_params}}}
        info["params"] = {}
        for d, p_d in enumerate(self.input_params):
            info["params"][p_d] = {"prior":
                {"min": self.bounds[d, 0], "max": self.bounds[d, 1]}}
        model = get_model(info)
        return model

class Spike(Model_generator):
    """
    A very pathological case where there's a gaussian with a very narrow gaussian (spike)
    in it. Although it is generalised to n dimensions be aware that it is only really
    supposd to be used in 2.
    If no bounds are provided they are set to (-4,4) in every dimension as standard.
    """

    def __init__(self, ndim=2, a=100., b=2., bounds=None, input_params=None):
        self.a = a
        self.b = b
        if bounds is None:
            self.bounds = np.array([[-4., 4.]]*ndim)
        else:
            self.bounds = bounds
        self.auto_generate_parameter_names(input_params, ndim)

    def get_model(self):
        def sp(x):
            return -np.log(np.exp(-x*x)+(1.-np.exp(-self.b*self.b))*np.exp(-self.a*(x-self.b)**2))
        def spike(**kwargs):
            X = [kwargs[p] for p in self.input_params]
            chi2 = 0
            for i in range(len(X)):
                chi2+=sp(X[i])
            return -0.5*chi2
        info = {"likelihood": {"spike":
            {"external": spike, "input_params": self.input_params}}}
        info["params"] = {}
        for d, p_d in enumerate(self.input_params):
            info["params"][p_d] = {"prior":
                {"min": self.bounds[d, 0], "max": self.bounds[d, 1]}}
        model = get_model(info)
        return model
