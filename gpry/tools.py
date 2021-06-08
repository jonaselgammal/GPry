"""
This module contains general tools used in different parts of the code.
"""

import numpy as np
from numpy import trace as tr
from numpy.linalg import det


def kl_norm(mean_0, cov_0, mean_1, cov_1):
    """
    Computes the KL divergence between two normal distributions defined by their means
    and covariance matrices.
    """
    cov_1_inv = np.linalg.inv(cov_1)
    dim = len(mean_0)
    return 0.5 * (np.log(det(cov_1)) - np.log(det(cov_0)) - dim + tr(cov_1_inv @ cov_0)
                  + (mean_1 - mean_0).T @ cov_1_inv @ (mean_1 - mean_0))
