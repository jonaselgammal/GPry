"""
Provides different methods for drawing initial points from the prior.
"""

from abc import ABCMeta, abstractmethod
import numpy as np

class Generator(metaclass=ABCMEta):
    """
    Abstract base class for all initial point generators. This implements
    some basic functionality and all generators should inherit from it.
    """
    def __init__(self, bounds):
        self.bounds = bounds

    @abstractmethod
    def generate(self, n_points, f):
        """Generate a fixed number of initial points from f. Tries to draw
        initial points until the desired number of (finite) points have
        been drawn. Finite and infinite locations and values are then
        returned to train the SVM and GP.
        """

# This module is not functional yet. Adding the functionality will be done as
# soon as the SVM and Convergence criterion have been updated.
